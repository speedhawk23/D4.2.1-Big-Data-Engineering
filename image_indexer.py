import os, sys, ctypes, sqlite3, hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

# Threads interner Libs drosseln (stabiler bei Parallel-I/O)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_NUM_THREADS", "1")

# Zentrale Konfiguration
@dataclass
class Config:
    DB_PATH: str = "images.db"
    IMAGE_FOLDER: str = r"D:/"
    BINS: Tuple[int, int] = (12, 12)       
    RESIZE: Tuple[int, int] = (100, 100) 
    BATCH_SIZE: int = 50_000
    UPDATE_BATCH_SIZE: int = 10_000  
    SCAN_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    FAST_DB_BULK: bool = True 
    NUM_WORKERS: int = os.cpu_count() or 4
    # Verzeichnisse/Dateien, die beim Scan ignoriert werden
    SKIP_COMPONENTS: Tuple[str, ...] = (
        '$recycle.bin', 'recycler', 'recycled',
        'system volume information', 'windows',
        'program files', 'program files (x86)',
        '.git', '__pycache__', '.trash', 'trash', '.ds_store'
    )

cfg = Config()
_SKIP_SET = set(cfg.SKIP_COMPONENTS)

def open_db(db_path: str, bulk: bool=False) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Performance-orientierte PRAGMAs (sicher genug für lokale DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-1048576")  # ~1 GiB Page Cache
    conn.execute(f"PRAGMA synchronous={'OFF' if (bulk and cfg.FAST_DB_BULK) else 'NORMAL'}")
    if bulk and cfg.FAST_DB_BULK:
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    return conn

def create_schema(conn: sqlite3.Connection) -> None:
    # Hinweis: KEIN embed_vector mehr in der Tabelle.
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            image_id     INTEGER PRIMARY KEY,
            path         TEXT UNIQUE NOT NULL,
            file_name    TEXT,
            directory    TEXT,
            file_size    INTEGER,
            mtime        REAL,
            hsv_vector   BLOB,
            hog_vector   BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_path ON images(path);
        CREATE INDEX IF NOT EXISTS idx_dir  ON images(directory);
        """)

def generate_image_id(path: str) -> int:
    # Stabiler 63-bit Key aus Pfad
    h = hashlib.blake2b(path.encode('utf-8'), digest_size=8).digest()
    x = int.from_bytes(h, 'big') & ((1 << 63) - 1)
    return x or 1

def _is_hidden_or_system(path: str) -> bool:
    base = os.path.basename(path)
    if sys.platform.startswith("win"):
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == -1:
                return False
            return bool(attrs & (0x2 | 0x4))  # HIDDEN | SYSTEM
        except Exception:
            return False
    return base.startswith('.')

def _should_skip_path(full_path: str) -> bool:
    low = os.path.normcase(full_path).lower()
    comps = [c for c in low.replace('/', os.sep).split(os.sep) if c]
    if any(c in _SKIP_SET for c in comps):
        return True
    return _is_hidden_or_system(full_path)

def _file_stat(p: str) -> Tuple[Optional[int], Optional[float]]:
    try:
        st = os.stat(p)
        return st.st_size, st.st_mtime
    except Exception:
        return None, None

def iter_image_files(folder: str, extensions=None) -> Iterable[str]:
    exts = set([e.lower() for e in (extensions or cfg.SCAN_EXTS)])
    for root, dirs, files in os.walk(folder, topdown=True):
        # Verzeichnisse vorab filtern (spart os.walk-Tiefe)
        dirs[:] = [d for d in dirs if not _should_skip_path(os.path.join(root, d))]
        if _should_skip_path(root):
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                full = os.path.abspath(os.path.join(root, f))
                if not _should_skip_path(full):
                    yield full

def _bulk_upsert(cur: sqlite3.Cursor, rows: List[Tuple[int,str,str,str,Optional[int],Optional[float]]]) -> None:
    # Neue Records einfügen
    cur.executemany(
        "INSERT OR IGNORE INTO images (image_id, path, file_name, directory, file_size, mtime) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows
    )
    # Metadaten geänderter Dateien aktualisieren + Feature-Vektoren invalidieren (embed_vector existiert nicht mehr)
    cur.executemany(
        "UPDATE images SET file_name=?, directory=?, file_size=?, mtime=?, "
        "hsv_vector=NULL, hog_vector=NULL "
        "WHERE path=? AND (COALESCE(file_size,-1) != COALESCE(?, -1) "
        "OR COALESCE(mtime,-1)  != COALESCE(?, -1))",
        [(fn, dr, sz, mt, p, sz, mt) for (_, p, fn, dr, sz, mt) in rows]
    )

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

def purge_recyclebin_entries(conn: sqlite3.Connection) -> int:
    patterns = ['%$recycle.bin%', '%recycler%', '%recycled%', '%system volume information%']
    where = " OR ".join(["LOWER(path) LIKE ?"] * len(patterns))
    with conn:
        cur = conn.execute(f"DELETE FROM images WHERE {where}", [p.lower() for p in patterns])
        return cur.rowcount

#CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["index","purge"], required=True)
    parser.add_argument("--db", default=cfg.DB_PATH)
    parser.add_argument("--folder", default=cfg.IMAGE_FOLDER)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ext", nargs="*", default=list(cfg.SCAN_EXTS))
    args = parser.parse_args()

    conn = open_db(args.db, bulk=True)

    if args.stage == "index":
        total = index_images_streaming(conn, args.folder, args.limit, set([e.lower() for e in args.ext]))
        print(f"Gescannt/indiziert: {total}")
    elif args.stage == "purge":
        n = purge_recyclebin_entries(conn)
        print(f"Gelöscht: {n} Einträge mit Recycle/System im Pfad")

    conn.close()
