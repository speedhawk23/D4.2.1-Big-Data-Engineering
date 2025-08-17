import os
import sys
import sqlite3
import numpy as np
import pytest

# Projekt-Root ins sys.path, damit die Imports auch von tests/ funktionieren
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import image_recommender_embeddings as emb


# Hilfsfunktionen für die Tests
def _mk_db(tmpdir):
    """legt eine kleine SQLite-DB für Tests an"""
    db_path = os.path.join(tmpdir, "images.db")
    conn = sqlite3.connect(db_path)
    try:
        emb.create_schema(conn)
    except Exception:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY, path TEXT NOT NULL)"
        )
    return conn, db_path

def _ins_images(conn, items):
    conn.executemany("INSERT INTO images(image_id, path) VALUES (?,?)", items)
    conn.commit()

def _root_for(db_path, outdir=None):
    return emb._feature_dir(db_path, outdir)

def _write_npy_for(iid, vec, db_path, shard_mod=16, shard_width=2, outdir=None):
    """schreibt ein Dummy-Embedding als NPY"""
    root = _root_for(db_path, outdir)
    p = emb._npy_path(root, int(iid), shard_mod, shard_width)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.asarray(vec, dtype=np.float32))
    return p


# -------------------------------------------------------------------
# Test: Paging und Erkennen fehlender Embeddings
# -------------------------------------------------------------------
def test_iter_pages_and_stream_missing_yields_expected_ids(tmp_path):
    conn, db_path = _mk_db(tmp_path)
    rows = [
        (1, str(tmp_path / "a.jpg")),
        (2, str(tmp_path / "b.jpg")),
        (3, str(tmp_path / "c.jpg")),
        (4, str(tmp_path / "d.jpg")),
        (5, str(tmp_path / "e.jpg")),
    ]
    _ins_images(conn, rows)

    # Nur für 1, 3 und 5 werden Embeddings abgelegt
    _write_npy_for(1, np.ones(4) / 2, db_path)
    _write_npy_for(3, np.array([1, 0, 0, 0]), db_path)
    _write_npy_for(5, np.array([0, 1, 0, 0]), db_path)

    root = _root_for(db_path, None)
    batches = list(
        emb.stream_missing_embeddings_fs(
            conn, page=3, root=root, shard_mod=16, shard_width=2, limit=None
        )
    )
    missing_ids = [iid for batch in batches for (iid, _) in batch]
    assert missing_ids == [2, 4]

    pages = list(emb._iter_image_id_pages(conn, page=2))
    assert len(pages) == 3  # ergibt 2/2/1
    assert pages[0][0][0] == 1 and pages[1][0][0] == 3 and pages[2][0][0] == 5
    conn.close()


# -------------------------------------------------------------------
# Test: Laden von NPYs (mit Normalisierung und Filter für schlechte Vektoren)
# -------------------------------------------------------------------
def test_load_all_embeds_skips_bad_vectors_and_normalizes(tmp_path):
    conn, db_path = _mk_db(tmp_path)
    _ins_images(
        conn,
        [
            (10, str(tmp_path / "x.jpg")),
            (11, str(tmp_path / "y.jpg")),
            (12, str(tmp_path / "z.jpg")),
        ],
    )

    # gültig, aber noch nicht normiert
    _write_npy_for(10, np.array([3.0, 4.0, 0.0, 0.0]), db_path)
    # Nullvektor -> sollte übersprungen werden
    _write_npy_for(11, np.zeros(4, dtype=np.float32), db_path)
    # NaN -> ebenfalls überspringen
    _write_npy_for(12, np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32), db_path)

    paths, M = emb._load_all_embeds_from_npy(
        conn, outdir=None, shard_mod=16, shard_width=2, limit=None
    )
    conn.close()

    # Es darf nur das eine gültige Bild drin sein
    assert paths == [str(tmp_path / "x.jpg")]
    assert M.shape == (1, 4)
    # Normierung sollte passiert sein
    n = float(np.linalg.norm(M[0]))
    assert 0.999 <= n <= 1.001