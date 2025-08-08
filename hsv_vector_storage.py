import sqlite3
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from typing import Optional, Tuple

# 🧠 HSV als NumPy-Array berechnen
def compute_hsv_vector(image_path: str, bins=(8, 8)) -> Optional[np.ndarray]:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        hsv = cv2.cvtColor(cv2.resize(image, (100, 100)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        print(f"❌ Fehler bei {image_path}: {e}")
        return None

# 🧵 Wrapper für Multiprocessing
def compute_hsv_safe(args: Tuple[int, str]) -> Optional[Tuple[int, bytes]]:
    image_id, path = args
    hsv = compute_hsv_vector(path)
    if hsv is not None:
        # Speichere HSV als Binärdaten (BLOB)
        return (image_id, hsv.tobytes())
    return None

# 🗃️ Hauptfunktion: HSV-Vektoren in SQLite als BLOB speichern
def update_hsv_vectors_parallel(db_path: str = "image_index.db", num_workers: int = 4, create_index=False):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(images)")
        columns = [col[1] for col in cursor.fetchall()]
        if "hsv_vector_blob" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN hsv_vector_blob BLOB")
            conn.commit()
        cursor.execute("SELECT image_id, path FROM images")
        images = cursor.fetchall()

    print(f"🎨 Parallel HSV mit {num_workers} Prozessoren...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(compute_hsv_safe, images, chunksize=100)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            batch = []
            for result in tqdm(results, total=len(images), desc="HSV-Berechnung"):
                if result:
                    batch.append(result)
                if len(batch) >= 1000:
                    cursor.executemany(
                        "UPDATE images SET hsv_vector_blob = ? WHERE image_id = ?", 
                        [(blob, img_id) for img_id, blob in batch]
                    )
                    conn.commit()
                    batch = []
            if batch:
                cursor.executemany(
                    "UPDATE images SET hsv_vector_blob = ? WHERE image_id = ?", 
                    [(blob, img_id) for img_id, blob in batch]
                )
                conn.commit()
            if create_index:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_hsv_blob ON images (hsv_vector_blob)")
                conn.commit()

if __name__ == "__main__":
    update_hsv_vectors_parallel("image_index.db", num_workers=4, create_index=False)