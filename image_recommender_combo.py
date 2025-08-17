import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from tqdm import tqdm
import time

# Projekt-Imports
from image_recommender_embeddings import _compute_query_vec, search_hnsw
from image_recommender_hsv import compute_hsv_vector as hsv_query_vec
from image_recommender_hsv import V_BINS as HSV_V_BINS
from image_indexer import open_db, cfg
from image_recommender_hog import _imread_color as hog_imread_color, _compute_hog_from_bgr as hog_query_vec

# Kleine Hilfsfunktionen
def _norm01(arr: np.ndarray) -> np.ndarray:
    # Bringt Werte robust auf [0,1]; bei Konstantsignal -> Nullen
    if arr.size == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32, copy=False)

def _unit(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    # Normiert Vektor auf Länge 1; ungültige Fälle -> None
    if v is None:
        return None
    v = v.astype(np.float32, copy=False)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return None
    return v / n

def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    # Cosinus-Ähnlichkeit (bei None -> 0)
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))

def _pad_hsv_if_needed(v: np.ndarray, target_dim: int) -> np.ndarray:
    # Falls ältere HSV-Vektoren (nur HS) in der DB liegen: mit Nullen (V-Hist) auffüllen
    if v.size == target_dim:
        return v
    hs_dim = int(cfg.BINS[0]) * int(cfg.BINS[1])
    if v.size == hs_dim and target_dim == hs_dim + int(HSV_V_BINS):
        z = np.zeros(int(HSV_V_BINS), dtype=v.dtype)
        return np.concatenate([v.astype(np.float32, copy=False), z])
    return v  # Fallback: unverändert

# Kombi-Suche
def search_combo(db_path: str,
                 query_path: str,
                 k: int = 10,
                 metrics: List[str] = ["embed","hsv","hog"],
                 weights: List[float] = [1,1,1],
                 candidates: int = 300,
                 outdir: Optional[str] = None,
                 bench: bool = False) -> List[Tuple[float,int,str,Tuple[float,float,float]]]:
    """
    Rückgabe: (final_score, image_id, path, (s_embed, s_hsv, s_hog))
    - candidates: Größe des HNSW-Vorkorbs (größer = genauer; kleiner = schneller)
    - bench: einfache Zeitmessung pro Schritt ausgeben
    """
    assert len(metrics) == len(weights), "metrics und weights müssen gleich lang sein."

    t0_total = time.perf_counter()

    # Query-Features bauen
    t0 = time.perf_counter()
    q_embed = _compute_query_vec(query_path)
    t1 = time.perf_counter()
    if q_embed is None:
        print("Query (Embeddings) fehlgeschlagen.")
        return []

    q_hsv = _unit(hsv_query_vec(query_path)) if ("hsv" in metrics) else None
    q_hog = _unit(hog_query_vec(hog_imread_color(query_path))) if ("hog" in metrics) else None
    t2 = time.perf_counter()

    # Kandidaten via HNSW (über alle Bilder)
    num_cands = max(k, candidates)
    embed_hits = search_hnsw(db_path, qvec=q_embed, k=num_cands, outdir=outdir)
    t3 = time.perf_counter()
    if not embed_hits:
        return []
    embed_hits = embed_hits[:num_cands]  # (cosine_dist, iid, path)
    cand_ids   = [int(iid) for _, iid, _ in embed_hits]
    cand_paths = [p for _, _, p in embed_hits]
    s_embed    = np.array([1.0 - float(d) for d,_,_ in embed_hits], dtype=np.float32)  # Distanz -> Score

    # HSV/HOG aus DB laden (keine Bilddateien anfassen)
    need_hsv = ("hsv" in metrics)
    need_hog = ("hog" in metrics)
    s_hsv = np.zeros(len(cand_ids), dtype=np.float32)
    s_hog = np.zeros(len(cand_ids), dtype=np.float32)

    conn = open_db(db_path)

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    target_hsv_dim = int(q_hsv.shape[0]) if q_hsv is not None else None

    t4 = time.perf_counter()
    for chunk in tqdm(list(_chunks(cand_ids, 800)), desc="Lade HSV/HOG aus DB", unit="chunk", dynamic_ncols=True):
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT image_id, hsv_vector, hog_vector FROM images WHERE image_id IN ({ph})",
            tuple(chunk)
        ).fetchall()
        idx_map: Dict[int, int] = {cid: i for i, cid in enumerate(cand_ids)}
        for iid, hsv_blob, hog_blob in rows:
            i = idx_map.get(int(iid))
            if i is None:
                continue
            if need_hsv and hsv_blob is not None and q_hsv is not None:
                v = np.frombuffer(hsv_blob, dtype=np.float32)
                if target_hsv_dim is not None:
                    v = _pad_hsv_if_needed(v, target_hsv_dim)
                v = _unit(v)
                s_hsv[i] = _cos(q_hsv, v)
            if need_hog and hog_blob is not None and q_hog is not None:
                v = np.frombuffer(hog_blob, dtype=np.float32)
                v = _unit(v)
                s_hog[i] = _cos(q_hog, v)
    conn.close()
    t5 = time.perf_counter()

    # Normierung pro Metrik
    s_embed_n = _norm01(s_embed)
    s_hsv_n   = _norm01(s_hsv) if need_hsv else np.zeros_like(s_embed_n)
    s_hog_n   = _norm01(s_hog) if need_hog else np.zeros_like(s_embed_n)
    t6 = time.perf_counter()

    # Gewichte anwenden und finalen Score berechnen
    m2w = {m:w for m,w in zip(metrics, weights)}
    w_embed = float(m2w.get("embed", 0.0))
    w_hsv   = float(m2w.get("hsv",   0.0))
    w_hog   = float(m2w.get("hog",   0.0))
    w_sum   = max(1e-8, (w_embed + w_hsv + w_hog))
    final   = (w_embed*s_embed_n + w_hsv*s_hsv_n + w_hog*s_hog_n) / w_sum

    order = np.argsort(-final)[:k]
    out = []
    for i in order:
        out.append((
            float(final[i]),
            int(cand_ids[i]),
            cand_paths[i],
            (float(s_embed_n[i]), float(s_hsv_n[i]), float(s_hog_n[i]))
        ))
    t7 = time.perf_counter()

    if bench:
        # Kurzer Überblick, was wie lange gebraucht hat
        print("\n[Benchmark]")
        print(f"  Embedding-Query-Vektor: { (t1 - t0)*1000:.1f} ms")
        print(f"  HSV/HOG-Query-Vektoren: { (t2 - t1)*1000:.1f} ms")
        print(f"  HNSW-Suche (Embeddings): { (t3 - t2)*1000:.1f} ms")
        print(f"  DB-Fetch + Cosine (HSV/HOG): { (t5 - t4)*1000:.1f} ms")
        print(f"  Normierung + Mischen: { (t7 - t6)*1000:.1f} ms")
        print(f"  Gesamt (alles): { (t7 - t0_total)*1000:.1f} ms\n")

    return out

# --------- CLI ---------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("search_combo (DB-basiertes Re-Ranking mit optionaler Benchmark)")
    parser.add_argument("--db", required=True, type=str)
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metrics", type=str, default="embed,hsv,hog",
                        help="z.B. embed,hsv,hog oder embed,hsv")
    parser.add_argument("--weights", type=str, default="1,1,1",
                        help="z.B. 2,1,1 (gleiche Länge wie --metrics)")
    parser.add_argument("--candidates", type=int, default=300,
                        help="Kandidaten aus EMBED/HNSW fürs Re-Ranking")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--bench", type=int, default=0,
                        help="1 = Laufzeiten pro Schritt ausgeben")
    args = parser.parse_args()

    mets = [m.strip() for m in args.metrics.split(",") if m.strip()]
    w = [float(x) for x in args.weights.split(",")]
    if len(mets) != len(w):
        raise SystemExit("Fehler: --metrics und --weights müssen gleich viele Einträge haben.")

    try:
        cv2.utils.logging.setLogLevel(cv2.utils.LOG_LEVEL_ERROR)
    except Exception:
        pass

    hits = search_combo(
        db_path=args.db,
        query_path=args.query,
        k=args.k,
        metrics=mets,
        weights=w,
        candidates=args.candidates,
        outdir=args.outdir,
        bench=bool(args.bench)
    )

    for score, iid, path, (se, sh, sg) in hits:
        print(f"{score:.4f}\tIID={iid}\t{path}\t[embed={se:.3f} hsv={sh:.3f} hog={sg:.3f}]")
