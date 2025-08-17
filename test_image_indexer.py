import os
import sys
import time
from pathlib import Path
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # Projektwurzel bestimmen (eine Ebene höher)
if ROOT not in sys.path:                                                # Falls Root nicht im Python-Pfad
    sys.path.insert(0, ROOT)                                            # → hinzufügen, damit Modul importiert werden kann

# Import aus deinem Modul
import image_indexer as idx


@pytest.fixture(autouse=True)
def _limit_threads_env(monkeypatch):
    # Kleine Bremse für BLAS/Threads — Tests sollen stabil sein, nicht die CPU rösten
    monkeypatch.setenv("OMP_NUM_THREADS", "1")                          # OpenMP Threads auf 1 beschränken
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")                     # OpenBLAS auf 1 Thread
    monkeypatch.setenv("MKL_NUM_THREADS", "1")                          # MKL auf 1 Thread
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")                      # numexpr auf 1 Thread
    monkeypatch.setenv("CV_NUM_THREADS", "1")                           # OpenCV intern auf 1 Thread


@pytest.fixture
def tmp_db(tmp_path):
    # Frische, temporäre SQLite-DB pro Test
    db_path = tmp_path / "images.db"                                    # Neuer Pfad für temporäre DB im Testverzeichnis
    conn = idx.open_db(str(db_path), bulk=True)                         # DB öffnen (Bulk-Modus für Inserts)
    idx.create_schema(conn)                                             # Schema anlegen, sonst Fehler bei späteren Operationen
    yield conn                                                          # Übergibt Connection an den Test
    conn.close()


@pytest.fixture 
def sample_tree(tmp_path): 
    """Kleiner Test-Dateibaum: erlaubt + geskippt, damit wir die Filterlogik prüfen können.""" 
    root = tmp_path / "D"                                 # Root-Verzeichnis für Testdaten
    
    # erlaubt 
    (root / "photos").mkdir(parents=True)                 # "photos"-Ordner erstellen
    (root / "photos" / "a.jpg").write_bytes(b"JPG")       # Erlaubte JPG-Datei
    (root / "photos" / "b.png").write_bytes(b"PNG")       # Erlaubte PNG-Datei
 
    # geskippt: versteckter Ordner (Unix/Allgemein) 
    (root / ".git").mkdir()                               # Versteckter .git-Ordner
    (root / ".git" / "hidden.jpg").write_bytes(b"x")      # Datei darin soll ignoriert werden
 
    # geskippt: typische Windows-Systemordner 
    (root / "System Volume Information").mkdir()          # Windows-Systemordner
    (root / "System Volume Information" / "sys.jpg").write_bytes(b"x")  # Datei darin soll ignoriert werden
 
    # nicht erlaubte Extension 
    (root / "photos" / "note.txt").write_text("nope")     # Textdatei → soll nicht auftauchen
 
    return root                                           # Fixture gibt das Root-Verzeichnis zurück
 
 
def test_iter_image_files_respects_ext_and_skip(sample_tree): 
    exts = {".jpg", ".png", ".webp", ".jpeg"}             # Erlaubte Bild-Endungen
    paths = list(idx.iter_image_files(str(sample_tree), exts))  # Funktion aufrufen und Ergebnisse sammeln
 
    # Erwartung: nur die beiden erlaubten Bilder 
    assert len(paths) == 2                                # Genau 2 Bilder erlaubt
    assert any(p.endswith("a.jpg") for p in paths)        # "a.jpg" muss enthalten sein
    assert any(p.endswith("b.png") for p in paths)        # "b.png" muss enthalten sein
 
    # Dinge, die NICHT auftauchen dürfen 
    assert not any("System Volume Information" in p for p in paths)  # Windows-Systemordner ignoriert
    assert not any(".git" in p for p in paths)                       # .git-Ordner ignoriert
    assert not any(p.endswith(".txt") for p in paths)                 # Nicht erlaubte Extension ignoriert


def test_generate_image_id_stable_and_nonzero():
    p1 = r"D:\data\photos\img1.jpg"
    p2 = r"D:\data\photos\img1.jpg"
    p3 = r"D:\data\photos\img2.jpg"

    id1 = idx.generate_image_id(p1)                      # ID für erstes Bild
    id2 = idx.generate_image_id(p2)                      # ID für identischen Pfad → gleiche ID erwartet
    id3 = idx.generate_image_id(p3)                      # ID für anderes Bild → andere ID erwartet

    # Gleiches rein => gleiches raus
    assert id1 == id2
    # Anderer Pfad => anderer Key
    assert id1 != id3
    # Kein Null-Wert
    assert id1 != 0 and id2 != 0 and id3 != 0


def test_index_images_streaming_inserts_and_is_idempotent(tmp_db, sample_tree):
    # Erster Lauf: sollte 2 Bilder finden
    total_1 = idx.index_images_streaming(tmp_db, str(sample_tree), limit=None, extensions=set(idx.cfg.SCAN_EXTS))
    assert total_1 >= 2                                                # Erwartung: mindestens 2 Bilder wurden gefunden

    c = tmp_db.cursor()
    n_rows = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert n_rows == 2, "Zwei Bilddateien sollten in der DB sein"       # Nach erstem Lauf: 2 Bilder in DB

    # Zweiter Lauf: keine Duplikate dank UNIQUE(path)
    total_2 = idx.index_images_streaming(tmp_db, str(sample_tree), limit=None, extensions=set(idx.cfg.SCAN_EXTS))
    n_rows2 = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert n_rows2 == 2, "Kein Duplikat nach erneutem Indexlauf"        # Trotz erneutem Lauf → immer noch 2 Bilder
    assert total_2 >= 2                                                 # Auch der zweite Lauf meldet mindestens 2 Dateien


def test_metadata_change_invalidates_feature_vectors(tmp_db, tmp_path):
    # Mini-Setup: eine Datei indexieren
    img_dir = tmp_path / "photos"                                   # Temporäres Unterverzeichnis für Testbilder
    img_dir.mkdir()                                                 # Ordner anlegen
    img = img_dir / "x.jpg"                                        # Pfad zum Testbild
    img.write_bytes(b"JPG")                                        # Dummy-Inhalt schreiben (keine echte JPG, reicht für Pfadtests)

    # Indexieren
    idx.index_images_streaming(tmp_db, str(tmp_path), limit=None, extensions=set(idx.cfg.SCAN_EXTS))  # Ersten Indexlauf starten
    cur = tmp_db.cursor()                                          # DB-Cursor für direkte Updates/Abfragen

    # So tun als wären Features schon drin
    cur.execute("UPDATE images SET hsv_vector = x'01', hog_vector = x'02' WHERE path = ?", (str(img),))  # Feature-Spalten künstlich setzen
    tmp_db.commit()                                                # Änderungen persistieren

    # mtime ändern => sollte beim nächsten Lauf invalidiert werden
    time.sleep(0.02)                                               # Kleiner Sleep wegen Timestamp-Auflösung (Windows/FS)
    os.utime(str(img), None)                                       # Änderungszeit (mtime) anfassen

    # Nochmal indexieren
    idx.index_images_streaming(tmp_db, str(tmp_path), limit=None, extensions=set(idx.cfg.SCAN_EXTS))  # Zweiter Lauf sollte invalidieren

    hsv, hog = cur.execute("SELECT hsv_vector, hog_vector FROM images WHERE path = ?", (str(img),)).fetchone()  # Feature-Spalten holen
    assert hsv is None and hog is None, "Nach Dateiänderung müssen HSV/HOG auf NULL gesetzt werden."  # Erwartung: invalidiert → NULL


def test_purge_recyclebin_entries(tmp_db):
    # Ein paar künstliche Pfade reinschieben
    with tmp_db:                                                                                   # Kontext: autocommit/rollback
        tmp_db.execute("INSERT OR IGNORE INTO images (image_id, path) VALUES (?, ?)",
                       (123, r"D:\$Recycle.Bin\foo.jpg"))                                          # Eintrag aus Papierkorb (soll gelöscht werden)
        tmp_db.execute("INSERT OR IGNORE INTO images (image_id, path) VALUES (?, ?)",
                       (456, r"D:\System Volume Information\bar.jpg"))                             # Systemordner (soll gelöscht werden)
        tmp_db.execute("INSERT OR IGNORE INTO images (image_id, path) VALUES (?, ?)",
                       (789, r"D:\photos\ok.jpg"))                                                 # Normaler Pfad (soll bleiben)

    deleted = idx.purge_recyclebin_entries(tmp_db)                                                 # Funktion soll „Müll“-Einträge entfernen
    assert deleted >= 2                                                                            # Mindestens zwei Löschungen erwartet

    rows = tmp_db.execute("SELECT path FROM images").fetchall()                                    # Verbleibende Pfade aus DB laden
    remaining = {r[0].lower() for r in rows}                                                       # Case-insensitive Vergleich
    assert r"d:\photos\ok.jpg" in remaining                                                        # Gültiger Pfad ist noch da
    assert not any("recycle" in r for r in remaining)                                              # Keine Recycle-Bin-Pfade mehr
    assert not any("system volume information" in r for r in remaining)