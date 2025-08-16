import os, sys, ctypes, sqlite3, hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

# Limits internal library threads to 1 per process to avoid CPU overload and improve stability in parallel image processing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_NUM_THREADS", "1")

# central configuration for image scanning and database settings
@dataclass
class Config:
    DB_PATH: str = "images.db"
    IMAGE_FOLDER: str = r"D:/"
    BINS: Tuple[int, int] = (12, 12)
    RESIZE: Tuple[int, int] = (100, 100)
    BATCH_SIZE: int = 50_000    # store images in DB in chunks of 50k for speed
    UPDATE_BATCH_SIZE: int = 10_000   # update modified images in DB in chunks of 10k
    SCAN_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    FAST_DB_BULK: bool = True  #It makes inserting large amounts of data much faster by reducing disk sync operations.
    NUM_WORKERS: int = os.cpu_count() or 4    # number of parallel worker processes
# folders/files to skip during scan
    SKIP_COMPONENTS: Tuple[str, ...] = (
        '$recycle.bin', 'recycler', 'recycled',
        'system volume information', 'windows',
        'program files', 'program files (x86)',
        '.git', '__pycache__', '.trash', 'trash', '.ds_store'
    )
cfg = Config()
_SKIP_SET = set(cfg.SKIP_COMPONENTS)

# opens SQLite DB with optimized settings for faster reads/writes
def open_db(db_path: str, bulk: bool=False) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA temp_store=MEMORY")  # store temp tables in RAM for speed
    conn.execute("PRAGMA cache_size=-1048576") # set ~1GB SQLite page cache in RAM for faster queries on large datasets
    conn.execute(f"PRAGMA synchronous={'OFF' if (bulk and cfg.FAST_DB_BULK) else 'NORMAL'}") # use NORMAL sync mode (safe enough, moderate speed) when not in bulk insert
    if bulk and cfg.FAST_DB_BULK:  # lock DB exclusively during bulk insert for maximum write speed
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    return conn 

# create images table and indexes if they don't exist
def create_schema(conn: sqlite3.Connection) -> None:
    with conn:  #uses the context manager of sqlite3.Connection
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            image_id     INTEGER PRIMARY KEY,
            path         TEXT UNIQUE NOT NULL,    # store file path as text, must be unique and not null
            file_name    TEXT,
            directory    TEXT,
            file_size    INTEGER,
            mtime        REAL,   # last file modification time (used to detect changes and avoid unnecessary reprocessing)
            hsv_vector   BLOB,  # store feature vectors as binary data for efficient storage and retrieval
            embed_vector BLOB,
            hog_vector   BLOB  )
        );
        CREATE INDEX IF NOT EXISTS idx_path ON images(path);   # create indexes on path and directory for faster lookups
        CREATE INDEX IF NOT EXISTS idx_dir  ON images(directory);
        """)

# generate a unique, stable 63-bit positive ID from the image path using blake2b hashing
def generate_image_id(path: str) -> int:
    h = hashlib.blake2b(path.encode('utf-8'), digest_size=8).digest()
    x = int.from_bytes(h, 'big') & ((1 << 63) - 1)
    return x or 1

# determine if a file or folder should be treated as hidden or system-protected
# uses Windows API attributes on Windows, and dot-prefix naming on Unix/Mac
def _is_hidden_or_system(path: str) -> bool:
    base = os.path.basename(path)
    if sys.platform.startswith("win"):
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == -1: return False
            return bool(attrs & (0x2 | 0x4))
        except Exception:
            return False
    return base.startswith('.')

# skip path if it matches skip list or is hidden/system
def _should_skip_path(full_path: str) -> bool:
    low = os.path.normcase(full_path).lower()  # Normalizes the path (on Windows, converts to lowercase for case-insensitive comparison).
    comps = [c for c in low.replace('/', os.sep).split(os.sep) if c] # Splits the path into individual folder/file names, regardless of whether / or \ is used.
    if any(c in _SKIP_SET for c in comps): # If any of these names are in the skip list (_SKIP_SET) skip.
        return True
    return _is_hidden_or_system(full_path) # If not in the skip list, check whether the path is still marked as hidden or system.

# return file size and last modification time, or None if unavailable
def _file_stat(p: str) -> Tuple[Optional[int], Optional[float]]:
    try:
        st = os.stat(p)
        return st.st_size, st.st_mtime
    except Exception:
        return None, None
    
# recursively yield absolute paths of allowed image files, skipping unwanted folders/files
def iter_image_files(folder: str, extensions=None) -> Iterable[str]:
    exts = set([e.lower() for e in (extensions or cfg.SCAN_EXTS)])
    for root, dirs, files in os.walk(folder, topdown=True):
        dirs[:] = [d for d in dirs if not _should_skip_path(os.path.join(root, d))]
        if _should_skip_path(root):
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                full = os.path.abspath(os.path.join(root, f))
                if not _should_skip_path(full):
                    yield full

# scan image files, collect their metadata, and insert or update them in the DB in configurable batch sizes
# uses file size and modification time to detect changes, returns total number of processed images
def index_images_streaming(conn: sqlite3.Connection, folder: str, limit: Optional[int], extensions: set) -> int:
    create_schema(conn)
    cur = conn.cursor()
    to_insert: List[Tuple[int, str, str, str, Optional[int], Optional[float]]] = []
    n_total = 0
    for p in iter_image_files(folder, extensions):
        n_total += 1
        size, mt = _file_stat(p)
        to_insert.append((generate_image_id(p), p, os.path.basename(p), os.path.dirname(p), size, mt))
        if len(to_insert) >= cfg.BATCH_SIZE:
            _bulk_upsert(cur, to_insert)
            conn.commit()
            to_insert.clear()
        if limit and n_total >= limit:
            break
    if to_insert:
        _bulk_upsert(cur, to_insert)
        conn.commit()
    return n_total

def _bulk_upsert(cur: sqlite3.Cursor, rows: List[Tuple[int,str,str,str,Optional[int],Optional[float]]]) -> None:
    # insert new image records or update changed ones in bulk
    cur.executemany(
        "INSERT OR IGNORE INTO images (image_id, path, file_name, directory, file_size, mtime) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows
    )
    # update metadata for changed files and reset feature vectors to NULL , uses COALESCE to compare even when DB values are NULL)
    cur.executemany(
        "UPDATE images SET file_name=?, directory=?, file_size=?, mtime=?, "
        "hsv_vector=NULL, embed_vector=NULL, hog_vector=NULL "
        "WHERE path=? AND (COALESCE(file_size,-1) != COALESCE(?, -1) "
        "OR COALESCE(mtime,-1)  != COALESCE(?, -1))",
        [(fn, dr, sz, mt, p, sz, mt) for (_, p, fn, dr, sz, mt) in rows]
    )

# delete all DB entries whose path matches known recycle bin or system folders
def purge_recyclebin_entries(conn: sqlite3.Connection) -> int:
    patterns = ['%$recycle.bin%', '%recycler%', '%recycled%', '%system volume information%']
    where = " OR ".join(["LOWER(path) LIKE ?"] * len(patterns))
    with conn:
        cur = conn.execute(f"DELETE FROM images WHERE {where}", [p.lower() for p in patterns])
        return cur.rowcount
    
# CLI entry point: parse arguments, run indexing or purge, then close the DB
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["index","purge"], required=True)
    parser.add_argument("--db", default=cfg.DB_PATH)
    parser.add_argument("--folder", default=cfg.IMAGE_FOLDER)
    parser.add_argument("--limit", type=int, default=None, help="Maximale Anzahl an Bildern zum Indizieren")
    parser.add_argument("--ext", nargs="*", default=list(cfg.SCAN_EXTS))
    args = parser.parse_args()

    conn = open_db(args.db, bulk=True)
    if args.stage == "index":
        total = index_images_streaming(
            conn,
            args.folder,
            args.limit,   #  limit
            set([e.lower() for e in args.ext])
        )
        print(f"Gescannt/indiziert: {total}")
    elif args.stage == "purge":
        n = purge_recyclebin_entries(conn)
        print(f"Gelöscht: {n} Einträge mit Recycle/System im Pfad")
    conn.close()

