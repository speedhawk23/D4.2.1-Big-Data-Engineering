import os, math, sqlite3, hashlib
from typing import List, Tuple, Optional, Iterable, Dict
import numpy as np
import cv2
from tqdm import tqdm
from image_indexer import cfg, open_db, create_schema
 
# setting theards for various libraries to 1 to avoid multithreading issues
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_NUM_THREADS", "1")
 
 
 
 
# ------------------------------------------------
# Backbone wird gebaut damit Klassifizierungen effizient durchgeführt werden können
# ------------------------------------------------
 
import torch
import torch.nn as nn
from torchvision import models
 
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (_DEVICE.type == "cuda")
_TORCH_DTYPE = torch.float16 if AMP else torch.float32
 
def _load_backbone():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) # EfficientNet-B0 ist das Deep-Learning-Modell
    m.classifier = nn.Identity()             # 1280-D Features ein vektor in einem 1280
    m.eval().to(device=_DEVICE, dtype=_TORCH_DTYPE)
    return m
 
_BACKBONE = _load_backbone()
 
# ------------------------------------------------
# Bild normalisierung und Vorverarbeitung (Schnelles Bild-Preprocessing)
# ------------------------------------------------
def _imread_rgb_fast(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
def _preprocess(path: str):
 
    arr = _imread_rgb_fast(path)
    if arr is None: return None
    arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA) # bild wird auf 224x224 pixel skaliert
    arr = arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2,0,1).contiguous()
    if _DEVICE.type == "cuda":
        t = t.pin_memory()
    return t.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True)
 
@torch.no_grad()
def _infer_batch(tensors: List[torch.Tensor]) -> np.ndarray:
    if not tensors:
        return np.empty((0,1280), np.float32)
    batch = torch.stack(tensors, 0)
    if AMP:
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16):
            feats = _BACKBONE(batch)
    else:
        feats = _BACKBONE(batch)
    feats = feats.float()
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
    return feats.cpu().numpy().astype(np.float32, copy=False)
 
def _vec_to_blob(v: np.ndarray) -> bytes: # vector DB-Platz sparen
    return v.astype(np.float16, copy=False).tobytes()
 
def _blob_to_vec(b: bytes) -> np.ndarray: # quantisierungfehler glätten
    v = np.frombuffer(b, dtype=np.float16).astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    return (v/(n+1e-8)).astype(np.float32, copy=False)
 
# ------------------------------------------------
# SQLite-Schnellschalter
# ------------------------------------------------  
def _apply_fast_pragmas(conn: sqlite3.Connection):
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-400000;")  # ~400 MB RAM-Cache (bei 16 GB ist ok)
        try:
            conn.execute("PRAGMA mmap_size=30000000000;")
        except Exception:
            pass
    except Exception:
        pass
 
def write_embeddings(conn: sqlite3.Connection, batch: List[Tuple[int, bytes]]) -> None:
    with conn:
        conn.executemany(
            "UPDATE images SET embed_vector=? WHERE image_id=?",
            [(blob, iid) for (iid, blob) in batch]
        )
 
def stream_missing_embeddings(conn: sqlite3.Connection, page:int) -> Iterable[List[Tuple[int,str]]]:
    last_id = 0
    while True:
        rows = conn.execute(
            "SELECT image_id, path FROM images "
            "WHERE embed_vector IS NULL AND image_id > ? "
            "ORDER BY image_id LIMIT ?",
            (last_id, page)
        ).fetchall()
        if not rows: break
        last_id = rows[-1][0]
        yield rows
 
# ------------------------------------------------
# Embedding-Pipeline
# ------------------------------------------------  
from concurrent.futures import ThreadPoolExecutor, as_completed
IO_WORKERS = int(os.environ.get("IMG_IO_WORKERS", "10")) # Threads fürs gleichzeitige Bild-Laden und den Preprocess.    
                                                         # 8–12 für 512GB SSD gut
 
def _preprocess_many(rows: List[Tuple[int,str]]): # parallele zeilenverarbeitung
                                                  # beschädigte bilder fliegen raus
    def work(r):
        iid, p = r
        t = _preprocess(p)
        return iid, t
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for f in as_completed(futs):
            iid, t = f.result()
            if t is not None:
                yield iid, t
 
@torch.no_grad()
def process_embeddings(conn: sqlite3.Connection, mini_batch:int=384, limit: Optional[int]=None) -> None:
    create_schema(conn)
    _apply_fast_pragmas(conn)
 
    ok = fail = 0
    processed_total = 0
    buf_ids: List[int] = []
    buf_tensors: List[torch.Tensor] = []
    out_batch: List[Tuple[int, bytes]] = []
 
    def flush_infer():
        nonlocal ok, out_batch, buf_ids, buf_tensors
        if not buf_tensors:
            return
        feats = _infer_batch(buf_tensors)
        for i, iid in enumerate(buf_ids):
            out_batch.append((iid, _vec_to_blob(feats[i])))
            if len(out_batch) >= cfg.UPDATE_BATCH_SIZE:
                write_embeddings(conn, out_batch)
                ok += len(out_batch)
                out_batch.clear()
        buf_ids.clear()
        buf_tensors.clear()
 
    # Seitenweise streamen; pro Seite eine Fortschrittsanzeige
    for page_rows in stream_missing_embeddings(conn, cfg.BATCH_SIZE):
        if limit is not None and processed_total >= limit:
            break
 
        take = len(page_rows) if (limit is None) else max(0, min(len(page_rows), limit - processed_total))
        if take == 0 and limit is not None:
            break
 
        produced = 0
        with tqdm(total=take, desc=f"Embedding [{processed_total}/{'' if limit is None else limit}]", unit="img") as pbar:
            for iid, t in _preprocess_many(page_rows[:take]):
                produced += 1
                buf_ids.append(iid)
                buf_tensors.append(t)
                pbar.update(1)
 
                if len(buf_tensors) >= mini_batch:
                    flush_infer()
 
        # Rest der Seite flushen
        flush_infer()
        if out_batch:
            write_embeddings(conn, out_batch)
            ok += len(out_batch)
            out_batch.clear()
 
        processed_total += take
        fail += (take - produced)  # beschädigte / nicht ladbare Bilder
 
    print(f"Embeddings fertig: {ok} ok, {fail} Fehler")
 
 
# ------------------------------------------------
# Exact Ram fallbeck: Wenn du keinen HNSW-Index hast oder die Datenmenge klein ist.
# ------------------------------------------------
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
 
def load_exact_index(db_path:str) -> EmbedIndexExact:
    paths, vecs = [], []
    conn = open_db(db_path); _apply_fast_pragmas(conn)
    cur = conn.cursor(); cur.execute("SELECT path, embed_vector FROM images WHERE embed_vector IS NOT NULL")
    while True:
        rows = cur.fetchmany(100000)
        if not rows: break
        for (p, b) in rows:
            paths.append(p)
            vecs.append(_blob_to_vec(b))
    conn.close()
    M = np.vstack(vecs).astype(np.float32)
    return EmbedIndexExact(paths=paths, mat=M)
 
# ------------------------------------------------
# HNSW-Funktionen: Embeddings aus DB laden, Index aufbauen/laden und schnelle Ähnlichkeitssuche durchführen
# ------------------------------------------------
import hnswlib
 
def _hnsw_base(db_path:str) -> str:
    return f"{db_path}.embed.hnsw"
 
def build_hnsw_index(db_path:str, dim:int=1280, M:int=32, ef_construction:int=200, save:bool=True):
    conn = open_db(db_path); _apply_fast_pragmas(conn)
    cur = conn.cursor()
    cur.execute("SELECT image_id, embed_vector FROM images WHERE embed_vector IS NOT NULL")
    ids, vecs = [], []
    while True:
        rows = cur.fetchmany(50000)
        if not rows: break
        for iid, blob in rows:
            v = _blob_to_vec(blob)
            ids.append(int(iid))
            vecs.append(v)
    conn.close()
    if not ids:
        raise ValueError("Keine Embeddings vorhanden.")
 
    ids = np.asarray(ids, dtype=np.int64)
    data = np.vstack(vecs).astype(np.float32, copy=False)   # bereits L2-normalisiert
 
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=data.shape[0], M=M, ef_construction=ef_construction, random_seed=42)
    index.add_items(data, ids)
    index.set_ef(100)  # Qualitäts-/Speed-Schieber für die Suche
 
    if save:
        base = _hnsw_base(db_path)
        index.save_index(base + ".bin")
        np.save(base + ".ids.npy", ids)
    return index
 
def load_hnsw_index(db_path:str, dim:int=1280, ef:int=100):
    base = _hnsw_base(db_path)
    index = hnswlib.Index(space='cosine', dim=dim)
    index.load_index(base + ".bin")
    index.set_ef(ef) # ids sind nur nützlich, wenn man Mapping braucht .. hier lesen wir Pfad per SQL
    return index
 
def search_hnsw(db_path:str, qvec:np.ndarray, k:int=5):
    idx = load_hnsw_index(db_path, dim=len(qvec), ef=100)
    labels, dists = idx.knn_query(qvec.reshape(1,-1).astype(np.float32), k=k)
    labels, dists = labels[0], dists[0]
    scores = (1.0 - dists).astype(float)  # cosine-score
 
    conn = open_db(db_path)
    res = []
    for lab, sc in zip(labels, scores):
        row = conn.execute("SELECT path FROM images WHERE image_id=?", (int(lab),)).fetchone()
        if row:
            res.append((row[0], float(sc)))
    conn.close()
    return res
 
# ------------------------------------------------
# Erzeugt ein normalisiertes Embedding für ein einzelnes Query-Bild
# ------------------------------------------------
@torch.no_grad()
def _compute_query_vec(path: str) -> Optional[np.ndarray]:
    t = _preprocess(path)
    if t is None: return None
    t = t.unsqueeze(0)
    if AMP:
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16):
            feat = _BACKBONE(t)
    else:
        feat = _BACKBONE(t)
    v = feat.float().squeeze(0).cpu().numpy().astype(np.float32)
    v /= (np.linalg.norm(v)+1e-8)
    return v
 
# ------------------------------------------------
# CLI-Einstiegspunkt: Führt je nach --stage Embedding-Berechnung, Indexaufbau oder Bildsuche aus
# ------------------------------------------------
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
    args = parser.parse_args()
 
    if args.stage == "embed":
        conn = open_db(args.db, bulk=True)
        create_schema(conn)
        _apply_fast_pragmas(conn)
        process_embeddings(conn, mini_batch=args.batch, limit=args.limit)
        conn.close()
 
    elif args.stage == "build_index_hnsw":
        build_hnsw_index(args.db, dim=1280, M=32, ef_construction=200, save=True)
        print("HNSW-Index built and saved.")
 
    elif args.stage == "search_hnsw":
        if not args.query:
            print("Bitte --query angeben."); exit(1)
        q = _compute_query_vec(args.query)
        if q is None:
            print("Query-Bild konnte nicht verarbeitet werden."); exit(1)
        hits = search_hnsw(args.db, q, k=args.k)
        for p, s in hits:
            print(f"{s:.4f}\t{p}")
 
    elif args.stage == "search_exact":
        if not args.query:
            print("Bitte --query angeben."); exit(1)
        q = _compute_query_vec(args.query)
        if q is None:
            print("Query-Bild konnte nicht verarbeitet werden."); exit(1)
        idx = load_exact_index(args.db)
        hits = idx.search(q, k=args.k)
        for p, s in hits:
            print(f"{s:.4f}\t{p}")