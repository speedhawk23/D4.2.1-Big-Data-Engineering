import os 
import sys 
import numpy as np 
import cv2 
import pytest 
 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))    # Projektwurzel (eine Ebene höher) bestimmen
if ROOT not in sys.path: 
    sys.path.insert(0, ROOT)                                             # Root zum Python-Pfad hinzufügen (für Imports)
 
# Modul-Imports 
try: 
    import image_indexer as idx                                          # Hauptmodul importieren (Produktivpfad)
except ModuleNotFoundError: 
    import image_indexer as idx                                          # Fallback (falls Umgebung seltsam aufgelöst wird)
 
import image_recommender_hsv as hsv                                      # Modul für HSV-basierte Empfehlung laden
 
 
def _tmp_img(path, color_bgr): 
    """Kleines Dummy-Bild schreiben (80x120), OpenCV-kompatibel.""" 
    img = np.full((80, 120, 3), color_bgr, dtype=np.uint8)               # Erzeuge ein einfarbiges Testbild (BGR)
    ok = cv2.imwrite(str(path), img)                                     # Bild als Datei speichern
    assert ok, f"Kann Testbild nicht schreiben: {path}"                  # Sicherstellen, dass Schreiben geklappt hat
 
 
@pytest.fixture(autouse=True) 
def _limit_threads_env(monkeypatch): 
    """Threads klein halten → stabilere Tests, weniger CPU-Rauschen.""" 
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", 
              "NUMEXPR_NUM_THREADS", "CV_NUM_THREADS"]: 
        monkeypatch.setenv(k, "1")                                       # Alle relevanten Libs auf Single-Thread setzen


@pytest.fixture 
def tmp_db_with_two_images(tmp_path, monkeypatch): 
    """ 
    Frische SQLite-DB + Schema + 2 Testbilder eintragen. 
    hsv_vector bleibt NULL, damit process_hsv() sie befüllt. 
    """ 
    db_path = tmp_path / "images.db"                                     # Pfad für temporäre SQLite-DB
    conn = idx.open_db(str(db_path), bulk=True)                          # DB öffnen im Bulk-Modus
    idx.create_schema(conn)                                              # Tabellen-Schema erstellen

    d = tmp_path / "imgs"                                                # Verzeichnis für Testbilder
    d.mkdir() 
    p1 = d / "a.jpg" 
    p2 = d / "b.jpg" 
    _tmp_img(p1, (10, 200, 20))    # grünlicher Ton                      # Erstes Testbild → grünlich eingefärbt
    _tmp_img(p2, (200, 10, 20))    # rötlicher Ton                       # Zweites Testbild → rötlich eingefärbt

    rows = [                                                             # Metadaten für beide Bilder vorbereiten
        (idx.generate_image_id(str(p1)), str(p1), p1.name, str(d), os.path.getsize(p1), os.path.getmtime(p1)), 
        (idx.generate_image_id(str(p2)), str(p2), p2.name, str(d), os.path.getsize(p2), os.path.getmtime(p2)), 
    ] 
    with conn:                                                           # Datenbank-Insert in einem Commit
        conn.executemany( 
            "INSERT OR IGNORE INTO images (image_id, path, file_name, directory, file_size, mtime) " 
            "VALUES (?, ?, ?, ?, ?, ?)", 
            rows 
        ) 

    # Für Tests FAISS aus (dann überall gleich) 
    monkeypatch.setattr(hsv, "USE_FAISS", False, raising=False)          # Falls Code FAISS nutzt → deaktivieren für Tests

    yield conn                                                           # Übergibt die geöffnete DB-Verbindung an Test
    conn.close()                                                         # Nach Test: Verbindung schließen


# ---------- Unit-Tests: reine HSV-Funktionalität ----------

def test_compute_hsv_vector_color_and_gray(tmp_path):
    """Grundcheck: Feature-Extraktion liefert Vektor in erwarteter Länge/Typ."""  # Smoke-Test für Dimension & dtype
    color = tmp_path / "c.png"                                   # temporärer Pfad für Farb-Bild
    gray = tmp_path / "g.png"                                    # temporärer Pfad für Graustufen-Bild
    _tmp_img(color, (10, 200, 20))     # Farbe                    # erzeugt Testbild mit spezifischem RGB
    _tmp_img(gray, (128, 128, 128))    # Grauton                 # erzeugt neutrales Grau-Testbild

    v1 = hsv.compute_hsv_vector(str(color))                      # berechnet HSV-Featurevektor für Farb-Bild
    v2 = hsv.compute_hsv_vector(str(gray))                       # berechnet HSV-Featurevektor für Grau-Bild

    assert v1 is not None and v2 is not None, "Feature-Extraktion darf nicht None liefern"  # Funktion muss liefern

    HS_DIM = hsv.cfg.BINS[0] * hsv.cfg.BINS[1]                   # HS-Histogramm-Dimension (H×S-Bins)
    TARGET_DIM = HS_DIM + hsv.V_BINS                             # Gesamtdimension inkl. V-Bins

    assert v1.shape[0] == TARGET_DIM                             # Länge des Vektors muss passen (Farbe)
    assert v2.shape[0] == TARGET_DIM                             # Länge des Vektors muss passen (Grau)
    # dtype: float16 optional, sonst float32 – beides okay
    assert v1.dtype in (np.float16, np.float32)                  # akzeptierte Datentypen prüfen
    assert v2.dtype in (np.float16, np.float32)                  # akzeptierte Datentypen prüfen


def test__compute_query_vec_is_unit_norm(tmp_path):
    """Query-Vektor muss L2-normalisiert und float32 sein."""     # prüft Normalisierung & Typ
    imgp = tmp_path / "q.png"                                     # Pfad für Query-Testbild
    _tmp_img(imgp, (40, 170, 220))                                # erzeugt Query-Bild mit RGB-Farbe

    q = hsv._compute_query_vec(str(imgp))                         # interner Query-Vektor (für Suche/Matching)
    assert q is not None                                          # darf nicht None sein
    assert q.dtype == np.float32                                  # Query-Vektor soll float32 sein
    n = np.linalg.norm(q)                                         # L2-Norm berechnen
    assert np.isfinite(n) and 0.999 <= n <= 1.001, f"Nicht normalisiert? Norm={n}"  # ~Einheitslänge erzwingen


# ---------- Integration: DB ← Features, Index laden, Suche ----------

def test_process_hsv_writes_blobs(tmp_db_with_two_images):
    """process_hsv() soll BLOBs in die DB schreiben (hsv_vector != NULL)."""   # Testet ob Feature-Extraktion wirklich DB-BLOBs speichert
    conn = tmp_db_with_two_images                                              # erzeugt Test-Datenbank mit 2 Bildern
    hsv.process_hsv(conn, limit=None)                                          # berechnet HSV-Features und schreibt sie in DB

    cur = conn.cursor()
    got = list(cur.execute("SELECT hsv_vector FROM images WHERE hsv_vector IS NOT NULL"))  # holt alle Bilder mit Feature-Vektor
    assert len(got) == 2, "Beide Testbilder sollten HSV-Features haben"        # beide Testbilder müssen Features haben
    blob = got[0][0]
    assert isinstance(blob, (bytes, bytearray))                                # gespeicherter Wert muss ein BLOB sein

    # Blob → float32-Vektor
    arr = hsv._vec_from_blob(blob)                                             # konvertiert BLOB wieder zurück in NumPy-Vektor
    HS_DIM = hsv.cfg.BINS[0] * hsv.cfg.BINS[1]                                 # Dimension H×S
    TARGET_DIM = HS_DIM + hsv.V_BINS                                           # Dimension inkl. Value-Bins
    # Akzeptiere Altfall (nur HS) oder neue Komplette (HS+V)
    assert arr.shape[0] in (HS_DIM, TARGET_DIM)                                # Länge des Vektors muss stimmen (beide Varianten ok)
    assert arr.dtype == np.float32                                             # Features sollen float32 sein


def test_load_hsv_index_and_search_self_hit(tmp_db_with_two_images, monkeypatch):
    """Index laden und Abfrage: Top-1 muss das Bild selbst sein (Self-Hit).""" # prüft Index-Aufbau + Suchfunktionalität
    conn = tmp_db_with_two_images
    hsv.process_hsv(conn, limit=None)                                          # zuerst Features für Bilder in DB erzeugen

    # Pfad sichern, dann Connection schließen -> verhindert SQLite-Lock
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]               # speichert den Pfad zur DB
    conn.close()                                                               # schließt DB, um Lock-Probleme zu vermeiden

    monkeypatch.setattr(hsv, "USE_FAISS", False, raising=False)                # schaltet FAISS-Nutzung aus → fallback Index
    index = hsv.load_hsv_index(db_path, persist=False)                         # lädt Index basierend auf DB-Inhalt
    assert isinstance(index.paths, list) and len(index.paths) == 2             # Index muss 2 Bildpfade enthalten

    p = index.paths[0]                                                         # nimmt den ersten Bildpfad
    q = hsv._compute_query_vec(p)                                              # erstellt Query-Vektor für dieses Bild
    res = index.search(q, k=2)                                                 # sucht die 2 ähnlichsten Bilder im Index

    assert isinstance(res, list) and len(res) == 2                             # Ergebnisliste mit 2 Treffern
    assert res[0][0] == p, "Bestes Match sollte das Query-Bild selbst sein"    # Top-1 Ergebnis = Self-Hit
    assert isinstance(res[0][1], float)                                        # Score/Distanz muss float sein



def test_find_similar_end_to_end_api(tmp_db_with_two_images, monkeypatch): 
    """Einmal komplett durch die öffentliche API."""                      # Integrationstest über die "offizielle" Schnittstelle
    conn = tmp_db_with_two_images                                        # Testdatenbank mit 2 Bildern
    hsv.process_hsv(conn, limit=None)                                    # berechnet und speichert HSV-Features für alle Bilder

    # Erst alles holen, dann Connection schließen → keine Locks 
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]         # DB-Pfad abfragen
    path = conn.execute("SELECT path FROM images ORDER BY image_id LIMIT 1").fetchone()[0]  # Pfad vom ersten Bild
    conn.close()                                                         # Verbindung schließen (sonst SQLite-Lock-Gefahr)

    monkeypatch.setattr(hsv, "USE_FAISS", False, raising=False)          # FAISS deaktivieren → nutzt Fallback-Index
    hits = hsv.find_similar(path, db_path, k=2, persist_index=False)     # Suche: ähnliche Bilder zum gegebenen Bildpfad
    assert len(hits) == 2                                                # genau 2 Treffer zurück
    assert hits[0][0] == path                                            # Self-Hit: bestes Match ist das Query-Bild selbst
    assert isinstance(hits[0][1], float)                                 # Score/Distanzwert muss float sein
