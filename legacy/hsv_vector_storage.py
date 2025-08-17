"""
LEGACY SCRIPT , nur für einmalige Migration alter Daten.
Dieses Skript schreibt HSV-Vektoren in die Spalte 'color_signature'
und wird vom neuen image_recommender_hsv.py NICHT benötigt.
Kompatibilität bleibt erhalten, da der neue Code COALESCE(hsv_vector, color_signature) nutzt.
"""

import sqlite3
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from typing import Optional, Tuple

# Farbprofil im HSV-Raum als NumPy-Vektor erstellen
def create_color_signature(img_path: str, bins: Tuple[int, int] = (8, 8)) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        hsv_img = cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv_img], [0, 1], None, bins, [0, 180, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        return histogram
    except Exception as err:
        print(f" Fehler bei Bild {img_path}: {err}")
        return None

# Multiprocessing-Wrapper
def process_color_signature(args: Tuple[int, str]) -> Optional[Tuple[int, bytes]]:
    img_id, img_path = args
    signature = create_color_signature(img_path)
    if signature is not None:
        return (img_id, signature.tobytes())  # Speichern als BLOB
    return None

# Farbprofile für alle Bilder in der Datenbank berechnen und speichern
def store_color_signatures(db_file: str = "images.db", workers: int = 4, build_index: bool = False):
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(images)")
        existing_cols = [col[1] for col in cur.fetchall()]
        if "color_signature" not in existing_cols:
            cur.execute("ALTER TABLE images ADD COLUMN color_signature BLOB")
            conn.commit()
        cur.execute("SELECT image_id, path FROM images")
        img_list = cur.fetchall()

    print(f" Berechne Farbprofile mit {workers} Prozessoren...")
    with mp.Pool(processes=workers) as pool:
        results = pool.imap_unordered(process_color_signature, img_list, chunksize=100)

        with sqlite3.connect(db_file) as conn:
            cur = conn.cursor()
            buffer = []
            for entry in tqdm(results, total=len(img_list), desc="Farbprofil-Berechnung"):
                if entry:
                    buffer.append(entry)
                if len(buffer) >= 1000:
                    cur.executemany(
                        "UPDATE images SET color_signature = ? WHERE image_id = ?",
                        [(blob, img_id) for img_id, blob in buffer]
                    )
                    conn.commit()
                    buffer = []
            if buffer:
                cur.executemany(
                    "UPDATE images SET color_signature = ? WHERE image_id = ?",
                    [(blob, img_id) for img_id, blob in buffer]
                )
                conn.commit()
            if build_index:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_color_signature ON images (color_signature)")
                conn.commit()

if __name__ == "__main__":
    store_color_signatures("images.db", workers=4, build_index=False)
