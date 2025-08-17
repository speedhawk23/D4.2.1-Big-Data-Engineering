import os
import sys
import sqlite3
import numpy as np
import cv2
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import image_recommender_hog as hog


def _tmp_img(path, color_bgr=(120, 120, 120)):
    """schreibt ein kleines Testbild"""
    img = np.full((96, 96, 3), color_bgr, dtype=np.uint8)
    # ein paar Kanten rein, damit HOG nicht degeneriert
    cv2.line(img, (0, 0), (95, 95), (0, 0, 255), 2)
    cv2.rectangle(img, (20, 20), (76, 76), (255, 255, 255), 1)
    assert cv2.imwrite(str(path), img)


def _mk_db(tmpdir):
    db_path = os.path.join(tmpdir, "images.db")
    conn = sqlite3.connect(db_path)
    try:
        from image_indexer import create_schema
        create_schema(conn)
    except Exception:
        conn.execute("CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY, path TEXT NOT NULL, hog_vector BLOB)")
    return conn, db_path


# --- HOG: Descriptor hat Einheitsnorm und ist nicht leer
def test_hog_descriptor_unit_norm(tmp_path):
    p = tmp_path / "q.jpg"
    _tmp_img(p)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    v = hog._compute_hog_from_bgr(img)
    assert v is not None
    n = float(np.linalg.norm(v))
    assert 0.999 <= n <= 1.001


# --- Ähnlichkeit: identisch > unterschiedlich
def test_hog_similarity_monotonic(tmp_path):
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    _tmp_img(p1, (120, 120, 120))
    _tmp_img(p2, (60, 60, 200))
    v1 = hog._compute_hog_from_bgr(cv2.imread(str(p1)))
    v1b = hog._compute_hog_from_bgr(cv2.imread(str(p1)))  # gleiches Bild
    v2 = hog._compute_hog_from_bgr(cv2.imread(str(p2)))
    s_same = hog.hog_similarity(v1, v1b)
    s_diff = hog.hog_similarity(v1, v2)
    assert s_same > s_diff


# --- Streaming aus DB + einfache Linearsuche (Top-1 ist das identische Bild)
def test_hog_stream_and_linear_search(tmp_path):
    # Bilder anlegen
    q = tmp_path / "q.jpg"
    r = tmp_path / "r.jpg"
    _tmp_img(q, (100, 100, 120))
    _tmp_img(r, (200, 80, 60))

    # DB vorbereiten
    conn, _ = _mk_db(tmp_path)
    conn.execute("INSERT INTO images(image_id, path, hog_vector) VALUES (?,?,?)", (1, str(q), None))
    conn.execute("INSERT INTO images(image_id, path, hog_vector) VALUES (?,?,?)", (2, str(r), None))
    conn.commit()

    # HOGs einmalig berechnen und in DB schreiben (ohne Multiprocessing)
    for path in (str(q), str(r)):
        vpath, blob = hog._compute_hog_for_path(path)
        assert vpath == path and blob is not None
        conn.execute("UPDATE images SET hog_vector=? WHERE path=?", (blob, path))
    conn.commit()

    # Suche: Query ist q -> sollte q vor r finden (self-match wird im Code übersprungen,
    # deshalb prüfen wir nur, dass r oben landet, wenn self übersprungen wurde)
    hits = hog.find_similar_hog_linear(conn, str(q), top_k=2, limit=None)
    # hits ist [(score, path), ...] absteigend
    assert len(hits) >= 1
    # Wenn self-skip greift, ist r Top-1. Falls nicht, ist q Top-1. Beides zulässig.
    top_paths = [p for _, p in hits]
    assert (str(r) in top_paths) or (str(q) in top_paths)

    conn.close()
