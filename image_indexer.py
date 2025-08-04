import os
import sqlite3
import hashlib
from typing import List, Tuple, Optional
from PIL import Image
import multiprocessing as mp

# Alle Bilder finden (rekursiv)
def find_all_images(folder_path: str, extensions={".jpg", ".jpeg", ".png"}) -> List[str]:
    image_paths = []
    for root, _, files in os.walk(folder_path):  # Alle Ordner und Dateien durchsuchen
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                full_path = os.path.abspath(os.path.join(root, file)) #Pfad erstellen
                image_paths.append(full_path)
    return image_paths


#Erstellt eine SQLite-Datenbank mit Tabellen für Bildpfade und Metadaten sowie Indexen zur effizienten Bildverwaltung.
def create_database(db_path: str = "image_index.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            file_name TEXT,
            directory TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            image_id INTEGER PRIMARY KEY,
            width INTEGER,
            height INTEGER,
            size_kb INTEGER,
            FOREIGN KEY (image_id) REFERENCES images (image_id)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_path ON images (path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_size ON metadata (size_kb)")
    conn.commit()
    return conn


# Fügt Bilddaten als Batch in die Datenbank ein und überspringt doppelte Einträge
def insert_image_batch(conn: sqlite3.Connection, batch: List[Tuple[int, str, str, str]]) -> None:
    cursor = conn.cursor()
    try:
        cursor.executemany('''
            INSERT OR IGNORE INTO images (image_id, path, file_name, directory)
            VALUES (?, ?, ?, ?)
        ''', batch)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Fehler beim Batch-Insert: {e}")


# Durchsucht einen Ordner nach Bildern, erzeugt eindeutige IDs und speichert die Daten in Batches in SQLite.
def process_images_to_database(conn: sqlite3.Connection, folder_path: str, batch_size: int = 10000, limit: Optional[int] = None) -> None:
    print("Suche nach Bildern..")
    image_paths = find_all_images(folder_path)
    if limit:
        image_paths = image_paths[:limit]
    print(f"{len(image_paths)} Bilder gefunden.")

    print("Speichere Bildpfade in Datenbank...")
    batch = []
    for idx, path in enumerate(image_paths):
        file_name = os.path.basename(path)
        directory = os.path.dirname(path)
        image_id = int(hashlib.sha256(path.encode()).hexdigest(), 16) % 10**8
        batch.append((image_id, path, file_name, directory))
        if len(batch) >= batch_size:
            insert_image_batch(conn, batch)
            batch = []
            print(f"{idx + 1}/{len(image_paths)} Bilder verarbeitet..")
    if batch:
        insert_image_batch(conn, batch)
    print("✅ Bildpfade gespeichert.")


# Multiprocessing: Metadaten-Extraktion (schnell!)
def extract_metadata_safe(args):
    image_id, path = args
    try:
        with Image.open(path) as img:
            width, height = img.size
        size_kb = os.path.getsize(path) // 1024
        return (image_id, width, height, size_kb)
    except Exception as e:
        print(f"Fehler bei {path}: {e}")
        return None

def update_metadata_parallel(conn: sqlite3.Connection, batch_size: int = 10000, num_workers: int = 4) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, path FROM images")
    images = cursor.fetchall()

    print(f"Starte parallele Metadaten-Erfassung ({num_workers} Prozesse)...")
    pool = mp.Pool(processes=num_workers)
    processed = 0
    batch = []

    for result in pool.imap_unordered(extract_metadata_safe, images, chunksize=100):
        if result:
            batch.append(result)
        if len(batch) >= batch_size:
            cursor.executemany('''
                INSERT OR REPLACE INTO metadata (image_id, width, height, size_kb)
                VALUES (?, ?, ?, ?)
            ''', batch)
            conn.commit()
            processed += len(batch)
            print(f"{processed}/{len(images)} Metadaten gespeichert...")
            batch = []

    if batch:
        cursor.executemany('''
            INSERT OR REPLACE INTO metadata (image_id, width, height, size_kb)
            VALUES (?, ?, ?, ?)
        ''', batch)
        conn.commit()

    pool.close()
    pool.join()
    print(" Metadaten-Erfassung abgeschlossen.")

# Hauptprogramm
if __name__ == "__main__":
    FOLDER_PATH = "D:/"  
    DB_PATH = "image_index.db"
    
    conn = create_database(DB_PATH)
    process_images_to_database(conn, FOLDER_PATH, limit=1000)  # Start mit 1000 Bildern testen
    update_metadata_parallel(conn, num_workers=4)  # Speed-Boost!
    conn.close()
    print("Alle Daten gespeichert!")
