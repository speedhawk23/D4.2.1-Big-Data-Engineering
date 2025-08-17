import os, math, sqlite3, hashlib, time
from typing import List, Tuple, Optional, Iterable, Dict
import numpy as np
import cv2
from tqdm import tqdm
from image_indexer import cfg, open_db, create_schema

# Thread-Limits (stabil + schnelle BLAS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_NUM_THREADS", "1")

# -----------------------------
# Backbone (EfficientNet-B0)
# -----------------------------
import torch
import torch.nn as nn
from torchvision import models

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (_DEVICE.type == "cuda")
_TORCH_DTYPE = torch.float16 if AMP else torch.float32

def _load_backbone():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    m.eval().to(device=_DEVICE, dtype=_TORCH_DTYPE)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return m

_BACKBONE = _load_backbone()

# -----------------------------
# NPY-Sharding
# -----------------------------
DEFAULT_SHARD_MOD   = 1024
DEFAULT_SHARD_WIDTH = 4
SAVE_DTYPE          = np.float16  # on-disk

def _feature_dir(db_path: str, outdir: Optional[str]) -> str:
    base = os.path.abspath(outdir) if outdir else os.path.abspath(os.path.dirname(db_path) or ".")
    root = os.path.join(base, "features", "embed")
    os.makedirs(root, exist_ok=True)
    return root

def _npy_path(root: str, image_id: int, shard_mod: int, shard_width: int) -> str:
    if shard_mod > 0:
        sub = f"{int(image_id) % int(shard_mod):0{int(shard_width)}d}"
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{int(image_id)}.npy")
    return os.path.join(root, f"{int(image_id)}.npy")

def _save_npy_atomic(final_path: str, arr: np.ndarray):
    d = os.path.dirname(final_path); os.makedirs(d, exist_ok=True)
    tmp = final_path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr.astype(SAVE_DTYPE, copy=False))
    os.replace(tmp, final_path)

def _exists_embed(root: str, iid: int, shard_mod: int, shard_width: int) -> bool:
    return os.path.exists(_npy_path(root, iid, shard_mod, shard_width))

# -----------------------------
# Bild-Preprocessing
# -----------------------------
def _imread_rgb_fast(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _preprocess(path: str):
    arr = _imread_rgb_fast(path)
    if arr is None: return None
    # INTER_AREA ist für Downsizing schnell + gut
    arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2,0,1).contiguous()
    return t  # CPU-Tensor

@torch.inference_mode()
def _infer_batch(tensors: List[torch.Tensor]) -> np.ndarray:
    if not tensors:
        return np.empty((0,1280), np.float32)
    batch = torch.stack(tensors, 0)  # CPU
    if _DEVICE.type == "cuda":
        try:
            batch = batch.pin_memory()
        except Exception:
            pass
        batch = batch.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True)
    if AMP and _DEVICE.type == "cuda":
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16):
            feats = _BACKBONE(batch)
    else:
        feats = _BACKBONE(batch)
    feats = feats.float()
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    return feats.cpu().numpy().astype(np.float32, copy=False)

# -----------------------------
# SQLite PRAGMAs (Read-Only schnell)
# -----------------------------
def _apply_fast_pragmas(conn: sqlite3.Connection):
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-400000;")
        try: conn.execute("PRAGMA mmap_size=30000000000;")
        except Exception: pass
    except Exception:
        pass

# -----------------------------
# DB-Iterators
# -----------------------------
def _iter_image_id_pages(conn: sqlite3.Connection, page: int) -> Iterable[List[Tuple[int,str]]]:
    last_id = 0
    while True:
        rows = conn.execute(
            "SELECT image_id, path FROM images WHERE image_id > ? ORDER BY image_id LIMIT ?",
            (last_id, page)
        ).fetchall()
        if not rows: break
        last_id = rows[-1][0]
        yield rows

def stream_missing_embeddings_fs(conn: sqlite3.Connection, page: int,
                                 root: str, shard_mod: int, shard_width: int,
                                 limit: Optional[int]=None) -> Iterable[List[Tuple[int,str]]]:
    emitted = 0
    for rows in _iter_image_id_pages(conn, page):
        todo = []
        for iid, p in rows:
            if not _exists_embed(root, int(iid), shard_mod, shard_width):
                todo.append((int(iid), p))
                emitted += 1
                if limit is not None and emitted >= limit:
                    break
        if todo:
            yield todo
        if limit is not None and emitted >= limit:
            break

# -----------------------------
# Exact RAM Index (aus NPYs)
# -----------------------------
class EmbedIndexExact:
    def __init__(self, paths: List[str], mat: np.ndarray):
        self.paths = paths
        self.mat = mat.astype(np.float32, copy=False)

    def search(self, q: np.ndarray, k:int=5):
        q = q.astype(np.float32); q /= (np.linalg.norm(q)+1e-8)
        k = min(k, len(self.paths))
        sims = self.mat @ q
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.paths[int(i)], float(sims[int(i)])) for i in idx]

def _load_all_embeds_from_npy(conn: sqlite3.Connection, outdir: Optional[str],
                              shard_mod:int, shard_width:int,
                              limit: Optional[int]=None) -> Tuple[List[str], np.ndarray]:
    root = _feature_dir(conn.execute("PRAGMA database_list").fetchone()[2], outdir)
    paths, vecs = [], []
    count = 0
    cur = conn.cursor()
    cur.execute("SELECT image_id, path FROM images ORDER BY image_id")
    while True:
        rows = cur.fetchmany(100000)
        if not rows: break
        for iid, p in rows:
            npy = _npy_path(root, int(iid), shard_mod, shard_width)
            if not os.path.exists(npy):
                continue
            v = np.load(npy, mmap_mode="r").astype(np.float32, copy=False)
            n = float(np.linalg.norm(v))
            if n == 0 or not np.isfinite(n):
                continue
            v = v / n
            paths.append(p)
            vecs.append(v)
            count += 1
            if limit is not None and count >= limit:
                break
        if limit is not None and count >= limit:
            break
    if not vecs:
        return [], np.zeros((0,), dtype=np.float32)
    return paths, np.vstack(vecs).astype(np.float32, copy=False)

def load_exact_index(db_path:str, outdir: Optional[str]=None,
                     shard_mod:int=DEFAULT_SHARD_MOD, shard_width:int=DEFAULT_SHARD_WIDTH) -> EmbedIndexExact:
    conn = open_db(db_path); _apply_fast_pragmas(conn)
    paths, M = _load_all_embeds_from_npy(conn, outdir, shard_mod, shard_width)
    conn.close()
    return EmbedIndexExact(paths=paths, mat=M)

# -----------------------------
# HNSW – Build/Load/Search
# -----------------------------
import hnswlib

def _hnsw_base(db_path:str, outdir: Optional[str]=None) -> str:
    base = os.path.abspath(outdir) if outdir else os.path.abspath(os.path.dirname(db_path) or ".")
    return os.path.join(base, "features", "embed", "hnsw")

def build_hnsw_index(db_path:str, dim:int=1280, M:int=32, ef_construction:int=200,
                     save:bool=True, outdir: Optional[str]=None,
                     shard_mod:int=DEFAULT_SHARD_MOD, shard_width:int=DEFAULT_SHARD_WIDTH,
                     limit: Optional[int]=None):
    conn = open_db(db_path); _apply_fast_pragmas(conn)
    ids, vecs = [], []
    root = _feature_dir(db_path, outdir)
    cur = conn.cursor(); cur.execute("SELECT image_id FROM images ORDER BY image_id")
    seen = 0
    with tqdm(desc="HNSW Build: load NPY", unit="img", dynamic_ncols=True) as pbar:
        while True:
            rows = cur.fetchmany(100000)
            if not rows: break
            for (iid,) in rows:
                npy = _npy_path(root, int(iid), shard_mod, shard_width)
                if not os.path.exists(npy):
                    continue
                v = np.load(npy, mmap_mode="r").astype(np.float32, copy=False)
                n = float(np.linalg.norm(v))
                if n == 0 or not np.isfinite(n):
                    continue
                v = v / n
                ids.append(int(iid))
                vecs.append(v)
                seen += 1
                pbar.update(1)
                if limit is not None and seen >= limit:
                    break
            if limit is not None and seen >= limit:
                break
    conn.close()
    if not ids:
        raise ValueError("Keine Embeddings (NPY) gefunden – bitte erst --stage embed ausführen.")

    ids = np.asarray(ids, dtype=np.int64)
    data = np.vstack(vecs).astype(np.float32, copy=False)

    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=data.shape[0], M=M, ef_construction=ef_construction, random_seed=42)
    index.add_items(data, ids)
    index.set_ef(100)

    if save:
        base = _hnsw_base(db_path, outdir)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        index.save_index(base + ".bin")
        np.save(base + ".ids.npy", ids)
    return index

def load_hnsw_index(db_path:str, dim:int=1280, ef:int=100, outdir: Optional[str]=None):
    base = _hnsw_base(db_path, outdir)
    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index(base + ".bin")
    index.set_ef(ef)
    return index

# -----------------------------
# Query-Vektor
# -----------------------------
@torch.inference_mode()
def _compute_query_vec(path: str) -> Optional[np.ndarray]:
    t = _preprocess(path)
    if t is None: return None
    t = t.unsqueeze(0)
    if AMP:
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16):
            feat = _BACKBONE(t.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True) if _DEVICE.type == "cuda" else t)
    else:
        feat = _BACKBONE(t.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True) if _DEVICE.type == "cuda" else t)
    v = feat.float().squeeze(0).cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v)+1e-8)
    return v

# -----------------------------
# HNSW-Suche
# -----------------------------
def search_hnsw(db_path:str, qvec: Optional[np.ndarray]=None, k:int=5,
                query_path: Optional[str]=None, outdir: Optional[str]=None) -> List[Tuple[float,int,str]]:
    if qvec is None:
        if not query_path:
            raise ValueError("Entweder qvec oder query_path angeben.")
        qvec = _compute_query_vec(query_path)
        if qvec is None:
            return []
    idx = load_hnsw_index(db_path, dim=len(qvec), ef=100, outdir=outdir)
    labels, dists = idx.knn_query(qvec.reshape(1,-1).astype(np.float32), k=k)
    labels, dists = labels[0], dists[0]
    conn = open_db(db_path)
    out: List[Tuple[float,int,str]] = []
    for lab, dist in zip(labels, dists):
        row = conn.execute("SELECT path FROM images WHERE image_id=?", (int(lab),)).fetchone()
        if row:
            out.append((float(dist), int(lab), row[0]))
    conn.close()
    out.sort(key=lambda x: x[0])
    return out

# -----------------------------
# EMBEDDING-PIPELINE (hoch performant, 1 Progressbar)
# -----------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
IO_WORKERS   = int(os.environ.get("IMG_IO_WORKERS", "24"))
QUEUE_MAX    = int(os.environ.get("IMG_QUEUE_MAX", "2048"))
FLUSH_SECS   = float(os.environ.get("IMG_FLUSH_SECS", "0.5"))

@torch.inference_mode()
def process_embeddings(conn: sqlite3.Connection, mini_batch:int=384, limit: Optional[int]=None,
                       outdir: Optional[str]=None, shard_mod:int=DEFAULT_SHARD_MOD,
                       shard_width:int=DEFAULT_SHARD_WIDTH, dtype: str="fp16") -> None:
    """
    Schreibt Embeddings NUR als NPY: outdir/features/embed/<shard>/<image_id>.npy
    DB wird NICHT mit Embeddings beschrieben.
    """
    import queue, threading

    global SAVE_DTYPE
    SAVE_DTYPE = np.float16 if dtype == "fp16" else np.float32

    create_schema(conn)
    _apply_fast_pragmas(conn)

    # Zielverzeichnis (Standard: neben DB)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    root = _feature_dir(db_path, outdir)

    # --- Pre-Scan: wie viele fehlen? (für saubere Progressbar)
    missing_ids = []
    cur = conn.cursor()
    cur.execute("SELECT image_id, path FROM images ORDER BY image_id")
    scanned = 0
    while True:
        rows = cur.fetchmany(200000)
        if not rows: break
        for iid, p in rows:
            if not _exists_embed(root, int(iid), shard_mod, shard_width):
                missing_ids.append((int(iid), p))
                if limit is not None and len(missing_ids) >= limit:
                    break
        if limit is not None and len(missing_ids) >= limit:
            break
        scanned += len(rows)

    total_missing = len(missing_ids)
    if total_missing == 0:
        print("Alles erledigt: keine fehlenden Embeddings gefunden.")
        return

    q: "queue.Queue[Tuple[int, Optional[torch.Tensor]]]" = queue.Queue(maxsize=QUEUE_MAX)
    stop_token = ( -1, None )

    def producer():
        # Parallel: I/O + Preprocess
        def work(item):
            iid, path = item
            t = _preprocess(path)
            return iid, t
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
            for iid, t in ex.map(work, missing_ids, chunksize=64):
                q.put((iid, t))
        q.put(stop_token)

    prod_th = threading.Thread(target=producer, daemon=True)
    prod_th.start()

    ok = fail = 0
    last_flush = time.time()
    batch_ids: List[int] = []
    batch_tensors: List[torch.Tensor] = []

    # TQDM: 1 Leiste über alle fehlenden
    with tqdm(total=total_missing, desc="Embeddings", unit="img",
              dynamic_ncols=True, smoothing=0.1) as pbar:

        cur_batch = mini_batch
        while True:
            iid, t = q.get()
            if (iid, t) == stop_token:
                # Rest flushen
                if batch_tensors:
                    try:
                        feats = _infer_batch(batch_tensors)
                        for i, bi in enumerate(batch_ids):
                            _save_npy_atomic(_npy_path(root, bi, shard_mod, shard_width), feats[i])
                            ok += 1
                        pbar.update(len(batch_ids))
                    except RuntimeError as e:
                        # OOM Fallback für den letzten Rest
                        if "CUDA" in str(e).upper() and len(batch_tensors) > 1:
                            cur_batch = max(1, len(batch_tensors)//2)
                            # split & retry
                            for s in range(0, len(batch_tensors), cur_batch):
                                sub_t = batch_tensors[s:s+cur_batch]
                                sub_i = batch_ids[s:s+cur_batch]
                                feats = _infer_batch(sub_t)
                                for i, bi in enumerate(sub_i):
                                    _save_npy_atomic(_npy_path(root, bi, shard_mod, shard_width), feats[i])
                                    ok += 1
                                pbar.update(len(sub_i))
                        else:
                            fail += len(batch_tensors)
                break

            if t is None:
                fail += 1
                pbar.update(1)
                continue

            batch_ids.append(iid)
            batch_tensors.append(t)

            # Zeitbasiertes Flush (verhindert Idle bei sehr langsamer Producer-Rate)
            need_time_flush = (time.time() - last_flush) >= FLUSH_SECS

            if len(batch_tensors) >= cur_batch or need_time_flush:
                try:
                    feats = _infer_batch(batch_tensors)
                    for i, bi in enumerate(batch_ids):
                        _save_npy_atomic(_npy_path(root, bi, shard_mod, shard_width), feats[i])
                        ok += 1
                    pbar.update(len(batch_ids))
                    # Durchsatz anzeigen
                    pbar.set_postfix_str(f"batch={cur_batch}, ok={ok}, fail={fail}")
                    batch_ids.clear()
                    batch_tensors.clear()
                    last_flush = time.time()
                    # nach Erfolg: Batch wieder anheben (sanft)
                    if cur_batch < mini_batch:
                        cur_batch = min(mini_batch, max(1, cur_batch*2))
                except RuntimeError as e:
                    # CUDA OOM -> Batch halbieren und sofort neu versuchen
                    if "CUDA" in str(e).upper() and cur_batch > 1:
                        cur_batch = max(1, cur_batch // 2)
                        # nichts schreiben, direkt mit kleinerem batch weitermachen
                    else:
                        # irreparabler Fehler -> diese Items zählen als fail
                        fail += len(batch_tensors)
                        pbar.update(len(batch_tensors))
                        batch_ids.clear()
                        batch_tensors.clear()
                        last_flush = time.time()

    print(f"Embeddings (NPY) fertig: {ok} ok, {fail} Fehler")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",
        choices=["embed", "build_index_hnsw", "search_hnsw", "search_exact"],
        required=True)
    parser.add_argument("--db", default=cfg.DB_PATH)
    parser.add_argument("--query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--shard_mod", type=int, default=DEFAULT_SHARD_MOD)
    parser.add_argument("--shard_width", type=int, default=DEFAULT_SHARD_WIDTH)
    parser.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")
    args = parser.parse_args()

    if args.stage == "embed":
        conn = open_db(args.db, bulk=True)
        create_schema(conn)
        _apply_fast_pragmas(conn)
        process_embeddings(
            conn, mini_batch=args.batch, limit=args.limit,
            outdir=args.outdir, shard_mod=args.shard_mod,
            shard_width=args.shard_width, dtype=args.dtype
        )
        conn.close()

    elif args.stage == "build_index_hnsw":
        build_hnsw_index(
            args.db, dim=1280, M=32, ef_construction=200, save=True,
            outdir=args.outdir, shard_mod=args.shard_mod,
            shard_width=args.shard_width, limit=args.limit
        )
        print("HNSW-Index built and saved.")

    elif args.stage == "search_hnsw":
        if not args.query:
            print("Bitte --query angeben."); exit(1)
        q = _compute_query_vec(args.query)
        if q is None:
            print("Query-Bild konnte nicht verarbeitet werden."); exit(1)
        hits = search_hnsw(args.db, qvec=q, k=args.k, outdir=args.outdir)
        for dist, iid, p in hits:
            score = 1.0 - dist
            print(f"{score:.4f}\t{iid}\t{p}")

    elif args.stage == "search_exact":
        if not args.query:
            print("Bitte --query angeben."); exit(1)
        q = _compute_query_vec(args.query)
        if q is None:
            print("Query-Bild konnte nicht verarbeitet werden."); exit(1)
        idx = load_exact_index(args.db, outdir=args.outdir, shard_mod=args.shard_mod, shard_width=args.shard_width)
        hits = idx.search(q, k=args.k)  # [(path, score)]
        for p, s in hits:
            print(f"{s:.4f}\t{p}")
