import os, time, random, sqlite3, math
import multiprocessing as mp
from typing import Optional, Tuple, Iterable, List
import numpy as np
import cv2
from tqdm import tqdm
from image_indexer import cfg, open_db, create_schema

# Constrain library thread pools to keep CPU utilization predictable under multiprocessing
# Avoid oversubscription (OpenBLAS/MKL can otherwise spawn many threads per process)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_NUM_THREADS", "1")

# Optional FAISS (ANN backend). Falls back to linear scan if not available.
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False  # keep code paths working without FAISS

# HOG parameters — fixed for a stable descriptor size across the whole corpus
HOG_WIN_SIZE = (64, 64)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

_hog_cpu = cv2.HOGDescriptor(
    HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE, HOG_CELL_SIZE, HOG_NBINS
)

# CUDA preproc (only grayscale+resize on GPU; HOG stays on CPU) 
def _cuda_available() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False  # NOTE: guard against OpenCV builds without CUDA

def _preprocess_gray_gpu(img_bgr: np.ndarray) -> Optional[np.ndarray]:
     #evaluate transfer overhead vs CPU for small 64x64 targets
    try:
        gpumat = cv2.cuda_GpuMat()
        gpumat.upload(img_bgr)
        gray = cv2.cuda.cvtColor(gpumat, cv2.COLOR_BGR2GRAY)
        resized = cv2.cuda.resize(gray, HOG_WIN_SIZE, interpolation=cv2.INTER_AREA)
        return resized.download()
    except Exception:
        # Some OpenCV+CUDA builds can throw on tiny frames; fall back to CPU path
        return None

def _preprocess_gray_cpu(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, HOG_WIN_SIZE, interpolation=cv2.INTER_AREA)

_GPU_FLAG = False  # set once from CLI

def _worker_init(gpu_flag: bool):
    # Runs once per worker; avoid per-task overhead here
    global _GPU_FLAG
    _GPU_FLAG = gpu_flag and _cuda_available()
    try: cv2.setNumThreads(0)  
    except: pass
    try:
        np.random.seed((os.getpid() * int(time.time())) % 1234567)
        random.seed(os.getpid())
    except: pass

def _imread_color(path: str) -> Optional[np.ndarray]:
    # Keep it simple; Pillow fallback not needed unless you hit exotic formats
    return cv2.imread(path, cv2.IMREAD_COLOR)

def _compute_hog_from_bgr(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    # Returns L2-normalized descriptor ,unit length. Safe for cosine/IP.
    if img_bgr is None:
        return None
    if _GPU_FLAG:
        img = _preprocess_gray_gpu(img_bgr)
        if img is None:  # fallback if GPU preproc failed
            img = _preprocess_gray_cpu(img_bgr)
    else:
        img = _preprocess_gray_cpu(img_bgr)
    if img is None:
        return None
    desc = _hog_cpu.compute(img).flatten().astype(np.float32)
    n = np.linalg.norm(desc)
    if n > 0:
        desc /= n  # keep unit norm; avoids renormalizing in every similarity call
    return desc

# Indexing
def _compute_hog_for_path(image_path: str):
    # Returns (path, blob) suitable for SQLite UPDATE
    img_bgr = _imread_color(image_path)
    desc = _compute_hog_from_bgr(img_bgr)
    if desc is None:
        return None, None
    return image_path, desc.tobytes()

def _iter_fs_paths(folder: str, exts: set, limit: Optional[int]) -> Iterable[str]:
    # Streams filesystem paths with an optional hard limit
    count = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if limit is not None and count >= limit:
                return
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(root, f)
                count += 1
        if limit is not None and count >= limit:
            return

def index_images_hog(conn: sqlite3.Connection, folder: str,
                     limit: Optional[int] = None,
                     exts: Optional[set] = None,
                     batch_size: int = getattr(cfg, "BATCH_SIZE", 50_000),
                     gpu: bool = False,
                     pool_chunksize: int = 256) -> int:
    """
    Populate images.hog_vector where it is still NULL.
    Primary source: DB scan; fallback: filesystem scan. Both paths respect `limit`.
    """
    if exts is None:
        exts = set(getattr(cfg, "SCAN_EXTS", {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}))

    # Prefer DB rows with missing HOG ,cheaper than scanning the FS again
    paths: List[str] = []
    try:
        if limit:
            cur = conn.execute(
                "SELECT path FROM images WHERE hog_vector IS NULL LIMIT ?",
                (int(limit),)
            )
        else:
            cur = conn.execute("SELECT path FROM images WHERE hog_vector IS NULL")
        paths = [row[0] for row in cur.fetchall()]
    except Exception:
        # if the table doesn't exist yet, fallback below will still make progress
        paths = []

    # Fallback: filesystem scan
    if not paths:
        paths = list(_iter_fs_paths(folder, exts, limit))

    if limit is not None and len(paths) > limit:
        paths = paths[:limit]

    if not paths:
        return 0

    total_indexed = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(ctx.cpu_count(), initializer=_worker_init, initargs=(gpu,)) as pool:
        iterator = pool.imap_unordered(_compute_hog_for_path, paths, chunksize=pool_chunksize)
        batch = []
        desc_text = "HOG Indexing [GPU-preproc]" if (gpu and _cuda_available()) else "HOG Indexing [CPU]"
        for path, descriptor in tqdm(iterator, total=len(paths), desc=desc_text):
            if path is None or descriptor is None:
                continue  # skip unreadable images
            batch.append((descriptor, path))
            if len(batch) >= batch_size:
                # Batch DB writes to amortize SQLite commit overhead
                conn.executemany("UPDATE images SET hog_vector=? WHERE path=?", batch)
                conn.commit()
                total_indexed += len(batch)
                batch.clear()
        if batch:
            conn.executemany("UPDATE images SET hog_vector=? WHERE path=?", batch)
            conn.commit()
            total_indexed += len(batch)
    return total_indexed

# Similarity 
def hog_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # Unit-norm vectors  , dot product equals cosine similarity (fast path)
    return float(np.dot(vec1, vec2))

# Streaming loader
def _load_all_hog_stream(conn: sqlite3.Connection, max_rows: Optional[int] = None) -> Iterable[Tuple[str, np.ndarray]]:
    """
    Yields (path, hog_vector) in chunks. Keeps memory steady during full corpus scans.
    NOTE: Avoid issuing other queries on the same cursor while iterating.
    """
    cur = conn.execute("SELECT path, hog_vector FROM images WHERE hog_vector IS NOT NULL")
    seen = 0
    chunk = 8192  # adjust if you see pressure on I/O
    while True:
        rows = cur.fetchmany(chunk)
        if not rows:
            break
        for path, vec_blob in rows:
            yield path, np.frombuffer(vec_blob, dtype=np.float32)
            seen += 1
            if max_rows is not None and seen >= max_rows:
                return

# Linear search (Top-K via min-heap)
def find_similar_hog_linear(conn: sqlite3.Connection, query_path: str, top_k: int = 5,
                            limit: Optional[int] = None):
    import heapq, os as _os
    q = _compute_hog_from_bgr(_imread_color(query_path))
    if q is None:
        return []
    heap: List[Tuple[float, str]] = []  # (score, path)
    for path, vec in _load_all_hog_stream(conn, max_rows=limit):
        # Skip self-match; different path spellings can still refer to the same file
        try:
            if _os.path.exists(path) and _os.path.exists(query_path) and _os.path.samefile(path, query_path):
                continue
        except Exception:
            if path == query_path:
                continue
        s = float(np.dot(q, vec))  # unit-norm , dot
        if len(heap) < top_k:
            heapq.heappush(heap, (s, path))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, path))
    # heap keeps O(n log ) behavior; final sort is only over k elements
    return sorted(heap, key=lambda x: x[0], reverse=True)

# FAISS helpers 
def _maybe_to_all_gpus(index):
    # spreads index across available GPUs if GPU build is present
    try:
        if hasattr(faiss, "StandardGpuResources"):
            return faiss.index_cpu_to_all_gpus(index)
    except Exception:
        pass
    return index

def _maybe_back_to_cpu(index):
    # Ensure we can always serialize even if construction used GPU memory
    try:
        return faiss.index_gpu_to_cpu(index)
    except Exception:
        return index

# FAISS index build streaming 
def build_faiss_from_db(conn: sqlite3.Connection,
                        index_path: str = "hog.faiss",
                        ids_path: str = "hog_ids.npy",
                        use_ivf: bool = True,
                        nlist: int = 4096,
                        train_sample_size: int = 200_000,
                        add_batch: int = 50_000,
                        limit: Optional[int] = None) -> int:
    """
    Memory-friendly two-pass build:
      1) Sample for training (reservoir) + determine dim
      2) Train IVF (if enabled) or use flat IP
      3) Stream-add normalized vectors in batches and store path array alongside
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not installed.")

    rng = np.random.default_rng(42)  # stable training behavior across runs
    sample: List[np.ndarray] = []
    sample_count = 0
    dim: Optional[int] = None

    def reservoir_add(v: np.ndarray):
        nonlocal sample_count, sample
        if len(sample) < train_sample_size:
            sample.append(v.copy())
        else:
            j = rng.integers(0, sample_count + 1)
            if j < train_sample_size:
                sample[j] = v.copy()
        sample_count += 1

    # training sample
    iter1 = _load_all_hog_stream(conn, max_rows=limit)
    for _, vec in tqdm(iter1, desc="FAISS Pass 1 (sample/train)"):
        if dim is None:
            dim = vec.shape[0]
        n = np.linalg.norm(vec)
        if n > 0:
            vec = (vec / n).astype(np.float32)
        else:
            continue  # skip degenerate vectors
        if use_ivf:
            reservoir_add(vec)

    if dim is None:
        raise RuntimeError("No HOG vectors found in DB.")

    # Build index
    if use_ivf:
        # IVF Flat with IP works well for cosine when inputs are unit-norm
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
        if len(sample) == 0:
            raise RuntimeError("Empty training sample for IVF.")
        Xtr = np.vstack(sample).astype(np.float32)
        index.train(Xtr)
    else:
        index = faiss.IndexFlatIP(dim)

    index = _maybe_to_all_gpus(index)

    # add vectors in batches
    paths: List[str] = []
    batch_vecs: List[np.ndarray] = []
    added = 0
    iter2 = _load_all_hog_stream(conn, max_rows=limit)
    for path, vec in tqdm(iter2, desc="FAISS Pass 2 (add/index)"):
        n = np.linalg.norm(vec)
        if n == 0:
            continue
        v = (vec / n).astype(np.float32)
        batch_vecs.append(v)
        paths.append(path)
        if len(batch_vecs) >= add_batch:
            X = np.vstack(batch_vecs)
            index.add(X)
            added += len(batch_vecs)
            batch_vecs.clear()
    if batch_vecs:
        X = np.vstack(batch_vecs)
        index.add(X)
        added += len(batch_vecs)

    # Serialize index + ids paths , keep in sync for search
    index = _maybe_back_to_cpu(index)
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(paths, dtype=object))
    return added

# FAISS search
def search_faiss(query_path: str,
                 index_path: str = "hog.faiss",
                 ids_path: str = "hog_ids.npy",
                 top_k: int = 5):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not installed.")
    if not os.path.exists(index_path) or not os.path.exists(ids_path):
        # guide user to build step rather than crash with obscure FAISS error
        raise FileNotFoundError("Index or ID file missing. Run --stage build_faiss first.")
    index = faiss.read_index(index_path)
    index = _maybe_to_all_gpus(index)

    ids = np.load(ids_path, allow_pickle=True)
    q = _compute_hog_from_bgr(_imread_color(query_path))
    if q is None:
        return []

    # Normalize query; FAISS index expects unit-norm for IP=cosine
    q = (q / (np.linalg.norm(q) + 1e-8)).astype(np.float32)
    D, I = index.search(q.reshape(1, -1), int(top_k))
    I = I[0]; D = D[0]
    out = []
    for score, idx in zip(D, I):
        if 0 <= idx < len(ids):
            out.append((float(score), str(ids[idx])))
    # D is already in descending order from FAISS, but keep sort for safety
    out.sort(key=lambda x: x[0], reverse=True)
    return out

# CLI 
if __name__ == "__main__":
    import argparse, sys

    # ---- Stage-Aliase: hog -> index, search -> search_linear ----
    alias_argv = sys.argv[1:].copy()
    for i, a in enumerate(alias_argv):
        if a == "--stage" and i + 1 < len(alias_argv):
            if alias_argv[i + 1] == "hog":
                alias_argv[i + 1] = "index"
            elif alias_argv[i + 1] == "search":
                alias_argv[i + 1] = "search_linear"

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["index", "search_linear", "build_faiss", "search_faiss"], required=True)
    parser.add_argument("--db", default=cfg.DB_PATH)
    parser.add_argument("--folder", default=cfg.IMAGE_FOLDER)
    parser.add_argument("--limit", type=int, default=None, help="Max images/vectors to process")
    parser.add_argument("--ext", nargs="*", default=list(getattr(cfg, "SCAN_EXTS", {".jpg",".jpeg",".png",".bmp",".tiff"})))
    parser.add_argument("--query", type=str)
    parser.add_argument("--gpu", type=int, default=0, help="1=enable CUDA preproc if available, 0=CPU")
    parser.add_argument("--topk", type=int, default=5)
    # FAISS params
    parser.add_argument("--faiss_index", default="hog.faiss")
    parser.add_argument("--faiss_ids", default="hog_ids.npy")
    parser.add_argument("--faiss_ivf", type=int, default=1)
    parser.add_argument("--faiss_nlist", type=int, default=4096)
    parser.add_argument("--faiss_train", type=int, default=200_000, help="Training sample size (reservoir)")
    parser.add_argument("--faiss_batch", type=int, default=50_000, help="Batch size for index.add()")
    args = parser.parse_args(alias_argv)

    gpu_flag = bool(args.gpu) and _cuda_available()

    # Open DB with bulk pragmas; create schema if missing
    conn = open_db(args.db, bulk=True)
    create_schema(conn)

    if args.stage == "index":
        total = index_images_hog(
            conn,
            args.folder,
            args.limit,
            set([e.lower() for e in args.ext]),
            batch_size=getattr(cfg, "BATCH_SIZE", 50_000),
            gpu=gpu_flag
        )
        print(f"Indexed (HOG): {total} [{'GPU-preproc' if gpu_flag else 'CPU'}]")

    elif args.stage == "search_linear":
        if not args.query:
            print("Missing --query"); sys.exit(1)
        res = find_similar_hog_linear(conn, args.query, top_k=args.topk, limit=args.limit)
        for s, p in res:
            print(f"{s:.4f} - {p}")

    elif args.stage == "build_faiss":
        n = build_faiss_from_db(
            conn,
            index_path=args.faiss_index,
            ids_path=args.faiss_ids,
            use_ivf=bool(args.faiss_ivf),
            nlist=int(args.faiss_nlist),
            train_sample_size=int(args.faiss_train),
            add_batch=int(args.faiss_batch),
            limit=args.limit
        )
        print(f"FAISS index built: {n} vectors -> {args.faiss_index}, IDs -> {args.faiss_ids}")

    elif args.stage == "search_faiss":
        if not args.query:
            print("Missing --query"); sys.exit(1)
        res = search_faiss(args.query, index_path=args.faiss_index, ids_path=args.faiss_ids, top_k=args.topk)
        for s, p in res:
            print(f"{s:.4f} - {p}")

    conn.close()