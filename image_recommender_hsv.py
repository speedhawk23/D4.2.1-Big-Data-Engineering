import os, time, random, multiprocessing as mp, sqlite3, hashlib
from typing import List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
from image_indexer import cfg, open_db, create_schema


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Configure FAISS and vector precision defaults with optional IVF parameters.
USE_FLOAT16 = True
USE_FAISS = FAISS_AVAILABLE
FAISS_USE_IVF = False
FAISS_NLIST = 1024
FAISS_NPROBE = 32

# Initialize worker processes by disabling OpenCV threads and setting unique random seeds.
def _worker_init():
    try: cv2.setNumThreads(0)
    except: pass
    try:
        np.random.seed((os.getpid() * int(time.time())) % 1234567)
        random.seed(os.getpid())
    except: pass

# Add image loader that first tries OpenCV, falls back to Pillow if needed.
def _imread_any(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

# Implement HSV feature extraction with masked color histogram, normalization, and compact vector output.  
def compute_hsv_vector(path: str) -> Optional[np.ndarray]:
    try:
        img = _imread_any(path)
        if img is None: return None
        img = cv2.resize(img, cfg.RESIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,32,32), (179,255,255))
        hist = cv2.calcHist([hsv], [0,1], mask, cfg.BINS, [0,180,0,256])
        hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L2)
        v = hist.flatten()
        v = v.astype(np.float16 if USE_FLOAT16 else np.float32, copy=False)
        return np.ascontiguousarray(v)
    except Exception:
        return None

# Add worker function to compute HSV vector and return as DB-storable bytes (BLOB)    
def process_feature_worker(args: Tuple[int, str]) -> Optional[Tuple[int, bytes]]:
    image_id, path = args
    v = compute_hsv_vector(path)
    if v is None: return None
    return (image_id, v.tobytes())

# Implement batch database update to store HSV vectors (BLOB) for multiple images.
def write_features(conn: sqlite3.Connection, batch: List[Tuple[int, bytes]]) -> None:
    with conn:
        conn.executemany(
            "UPDATE images SET hsv_vector = ? WHERE image_id = ?",
            [(blob, img_id) for (img_id, blob) in batch]
        )

# Process images in pages and in parallel, show progress, and save results to the database in batches while tracking successes and failures
def process_hsv(conn: sqlite3.Connection, limit: Optional[int] = None) -> None:
    page = cfg.BATCH_SIZE
    success = failed = 0
    last_id = 0
    processed_total = 0

    with mp.Pool(cfg.NUM_WORKERS, initializer=_worker_init) as pool:
        while True:
            # Limit-Stop
            if limit is not None and processed_total >= limit:
                break

            rows = conn.execute(
                "SELECT image_id, path FROM images "
                "WHERE hsv_vector IS NULL AND image_id > ? "
                "ORDER BY image_id LIMIT ?",
                (last_id, page)
            ).fetchall()
            if not rows:
                break

            last_id = rows[-1][0]

            if limit is not None:
                remaining = max(0, limit - processed_total)
                if remaining <= 0:
                    break
                rows = rows[:remaining]

            batch: List[Tuple[int, bytes]] = []
            for res in pool.imap_unordered(process_feature_worker, rows, chunksize=400):
                if res:
                    batch.append(res); success += 1
                    if len(batch) >= cfg.UPDATE_BATCH_SIZE:
                        write_features(conn, batch)
                        batch.clear()
                else:
                    failed += 1

            if batch:
                write_features(conn, batch)

            processed_total += len(rows)

    print(f"HSV-Fertig: {success} ok, {failed} Fehler (processed_total={processed_total})")


# Convert HSV vector from database BLOB to contiguous NumPy float32 array.
def _vec_from_blob(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=(np.float16 if USE_FLOAT16 else np.float32)).copy()
    return np.ascontiguousarray(arr, dtype=np.float32)

# The class links vectors ↔ file paths and searches for the top-k most similar images, either quickly using FAISS or simply via matrix multiplication in RAM.
class HSVIndex:
    def __init__(self, paths, mat=None, index=None, use_faiss=False):
        self.paths = paths; self.mat = mat; self.index = index; self.use_faiss = use_faiss
    def search(self, q: np.ndarray, k: int=5):
        q = q.astype(np.float32); q /= (np.linalg.norm(q)+1e-8)
        k = min(k, len(self.paths))
        if self.use_faiss and self.index is not None:
            D, I = self.index.search(q.reshape(1,-1), k)
            return [(self.paths[int(i)], float(D[0][j])) for j,i in enumerate(I[0])]
        sims = self.mat @ q
        idx = np.argpartition(-sims, k-1)[:k]; idx = idx[np.argsort(-sims[idx])]
        return [(self.paths[int(i)], float(sims[int(i)])) for i in idx]
    
# Add helper to get count of indexed images and maximum image_id from the database.
def _index_stats(conn: sqlite3.Connection) -> Tuple[int,int]:
    c = conn.execute("SELECT COUNT(*), COALESCE(MAX(image_id),0) FROM images WHERE hsv_vector IS NOT NULL").fetchone()
    return int(c[0]), int(c[1])

# Generate short SHA1 fingerprint for FAISS index based on image count, max ID, and vector dimension.
def _index_fingerprint(count:int, maxid:int, dim:int) -> str:
    return hashlib.sha1(f"{count}-{maxid}-{dim}-hsv".encode("utf-8")).hexdigest()[:12]

# Build FAISS index file path using database path and fingerprint.
def _faiss_path(db_path:str, fp:str) -> str:
    return f"{db_path}.hsv.faiss.{fp}.index"

# Save FAISS index to disk with fingerprint-based file naming.
def save_faiss_index(db_path: str, index, count:int, maxid:int, dim:int) -> Optional[str]:
    if not FAISS_AVAILABLE: return None
    fp = _index_fingerprint(count, maxid, dim)
    path = _faiss_path(db_path, fp)
    faiss.write_index(index, path)
    return path

def try_load_faiss_index(db_path:str, count:int, maxid:int, dim:int):
    if not FAISS_AVAILABLE: return None
    fp = _index_fingerprint(count, maxid, dim)
    path = _faiss_path(db_path, fp)
    if os.path.exists(path):
        return faiss.read_index(path)
    return None

def _compute_query_vec(path: str) -> Optional[np.ndarray]:
    v = compute_hsv_vector(path)
    if v is None: return None
    v = v.astype(np.float32, copy=False)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def load_hsv_index(db_path: str, persist: bool=False) -> HSVIndex:
    conn = open_db(db_path)
    count, maxid = _index_stats(conn)
    cur = conn.cursor()

    paths, vecs = [], []
    cur.execute("SELECT path, hsv_vector FROM images WHERE hsv_vector IS NOT NULL")
    while True:
        rows = cur.fetchmany(100_000)
        if not rows: break
        for path, blob in rows:
            if blob is None: continue
            v = _vec_from_blob(blob)
            if v.size != (cfg.BINS[0]*cfg.BINS[1]): continue
            n = np.linalg.norm(v)
            if n > 0: v = v / n
            paths.append(path); vecs.append(v)
    conn.close()
    if not vecs:
        raise ValueError("No HSV vectors found.")
    M = np.vstack(vecs).astype(np.float32)
    dim = M.shape[1]

    idx_loaded = try_load_faiss_index(db_path, count, maxid, dim) if USE_FAISS else None
    if idx_loaded is not None:
        return HSVIndex(paths, index=idx_loaded, use_faiss=True)
    
    
    if USE_FAISS and len(paths) >= 1:
        if FAISS_USE_IVF and M.shape[0] >= 10_000:
            quant = faiss.IndexFlatIP(dim)
            nlist = min(FAISS_NLIST, max(32, int(np.sqrt(M.shape[0]))))
            idx = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            samp = min(100_000, M.shape[0])
            rng = np.random.default_rng(42)
            idx.train(M[rng.choice(M.shape[0], size=samp, replace=False)])
            idx.add(M); idx.nprobe = min(FAISS_NPROBE, nlist)
        else:
            idx = faiss.IndexFlatIP(dim); idx.add(M)
        if persist:
            save_faiss_index(db_path, idx, len(paths), maxid, dim)
        return HSVIndex(paths, index=idx, use_faiss=True)
    

    return HSVIndex(paths, mat=M, use_faiss=False)
def find_similar(query_path: str, db_path: str, k: int=5, persist_index: bool=False):
    q = _compute_query_vec(query_path)
    if q is None: return []
    idx = load_hsv_index(db_path, persist=persist_index)
    return idx.search(q, k=k)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["hsv","search"], required=True)
    parser.add_argument("--db", default=cfg.DB_PATH)
    parser.add_argument("--query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--faiss_ivf", action="store_true")
    parser.add_argument("--persist_index", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.faiss_ivf:
        FAISS_USE_IVF = True

    if args.stage == "hsv":
        conn = open_db(args.db, bulk=True)
        create_schema(conn)
        process_hsv(conn, limit=args.limit)  # limit
        conn.close()
    elif args.stage == "search":
        if not args.query:
            print("Please specify --query.")
            raise SystemExit(1)
        hits = find_similar(args.query, args.db, k=args.k, persist_index=args.persist_index)
        for p, s in hits:
            print(f"{s:.4f}\t{p}")
