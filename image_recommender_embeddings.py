import os, math, sqlite3, hashlib, time
from typing import List, Tuple, Optional, Iterable, Dict
import numpy as np
import cv2
from tqdm import tqdm
from image_indexer import cfg, open_db, create_schema

# setting theards for various libraries to 1 to avoid multithreading issues
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CV_NUM_THREADS", "1")

# ------------------------------------------------
# Backbone wird gebaut damit Klassifizierungen effizient durchgeführt werden können -> EfficientNet-B0
# ------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = (_DEVICE.type == "cuda")
_TORCH_DTYPE = torch.float16 if AMP else torch.float32

def _load_backbone():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) # EfficientNet-B0 ist das Deep-Learning-Modell
    m.classifier = nn.Identity()
    m.eval().to(device=_DEVICE, dtype=_TORCH_DTYPE)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return m

_BACKBONE = _load_backbone()

# -----------------------------
# NPY-Sharding
# -----------------------------
DEFAULT_SHARD_MOD   = 1024                 # Standardwert für Sharding-Modulus
DEFAULT_SHARD_WIDTH = 4                    # Standardanzahl der Ziffern für Unterordnernamen
SAVE_DTYPE          = np.float16           # Datentyp, in dem die Arrays auf Festplatte gespeichert werden (kleiner Speicherbedarf)

def _feature_dir(db_path: str, outdir: Optional[str]) -> str:
    base = os.path.abspath(outdir) if outdir else os.path.abspath(os.path.dirname(db_path) or ".")  # Bestimmt Basisverzeichnis (entweder angegeben oder vom Datenbankpfad abgeleitet)
    root = os.path.join(base, "features", "embed")                                                # Baut Pfad für "features/embed"
    os.makedirs(root, exist_ok=True)                                                              # Erstellt Ordner falls nicht vorhanden
    return root                                                                                   # Gibt Pfad zurück

def _npy_path(root: str, image_id: int, shard_mod: int, shard_width: int) -> str:
    if shard_mod > 0:                                                                             # Falls Sharding aktiviert
        sub = f"{int(image_id) % int(shard_mod):0{int(shard_width)}d}"                            # Berechnet Unterordnername durch Restoperation (z. B. 0042)
        d = os.path.join(root, sub)                                                               # Baut Pfad für diesen Unterordner
        os.makedirs(d, exist_ok=True)                                                             # Erstellt Unterordner falls nicht vorhanden
        return os.path.join(d, f"{int(image_id)}.npy")                                            # Rückgabe: kompletter Pfad zur NPY-Datei
    return os.path.join(root, f"{int(image_id)}.npy")                                             # Ohne Sharding: Datei direkt im root-Ordner

def _save_npy_atomic(final_path: str, arr: np.ndarray):
    d = os.path.dirname(final_path); os.makedirs(d, exist_ok=True)                                # Stellt sicher, dass Zielordner existiert
    tmp = final_path + ".tmp"                                                                     # Erst wird temporäre Datei erstellt
    with open(tmp, "wb") as f:                                                                    # Öffnet temporäre Datei im Schreibmodus
        np.save(f, arr.astype(SAVE_DTYPE, copy=False))                                            # Speichert Array als NPY-Datei mit Ziel-Datentyp
    os.replace(tmp, final_path)                                                                   # Ersetzt die temporäre Datei atomar durch die endgültige (sicher gegen Abbrüche)

def _exists_embed(root: str, iid: int, shard_mod: int, shard_width: int) -> bool:
    return os.path.exists(_npy_path(root, iid, shard_mod, shard_width))                           # Prüft, ob die entsprechende NPY-Datei bereits existiert


# -----------------------------
# Bild-Preprocessing
# -----------------------------
def _imread_rgb_fast(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)                          # Liest Bild mit OpenCV im BGR-Format ein
    if img is None: return None                                       # Falls Bild nicht existiert oder kaputt ist → None zurück
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       # Konvertiert BGR → RGB (Standard für Machine Learning)

def _preprocess(path: str):
    arr = _imread_rgb_fast(path)                                      # Bild laden (RGB)
    if arr is None: return None                                       # Falls Laden fehlschlägt → None
    arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)   # Skaliert Bild auf 224x224 (INTER_AREA gut für Verkleinerung)
    arr = arr.astype(np.float32) / 255.0                              # Normiert Pixelwerte auf [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)          # Mean-Werte (ImageNet-Standard)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)          # Standardabweichung (ImageNet-Standard)
    arr = (arr - mean) / std                                          # Normalisierung: (x-mean)/std
    t = torch.from_numpy(arr).permute(2,0,1).contiguous()             # Umwandeln in Torch-Tensor (Reihenfolge: Channels zuerst)
    return t  # CPU-Tensor                                            # Rückgabe als Tensor (noch auf CPU)

@torch.inference_mode()
def _infer_batch(tensors: List[torch.Tensor]) -> np.ndarray:
    if not tensors:                                                   # Falls Liste leer → leeres Array zurück
        return np.empty((0,1280), np.float32)
    batch = torch.stack(tensors, 0)  # CPU                            # Stapelt mehrere Bilder zu einem Batch
    if _DEVICE.type == "cuda":                                        # Falls GPU verfügbar
        try:
            batch = batch.pin_memory()                                # "Pin Memory" für schnelleren Transfer zur GPU
        except Exception:
            pass
        batch = batch.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True)  # Transfer auf GPU (asynchron möglich)
    if AMP and _DEVICE.type == "cuda":                                # Falls Automatic Mixed Precision aktiviert
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16):      # Berechnung in float16 für Effizienz
            feats = _BACKBONE(batch)                                  # Feature-Extraktion durch Backbone-Netzwerk
    else:
        feats = _BACKBONE(batch)                                      # Normaler Vorwärtsdurchlauf ohne AMP
    feats = feats.float()                                             # Ergebnis wieder in float32
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)          # Normalisierung auf L2-Norm (jeder Vektor Länge ≈ 1)
    return feats.cpu().numpy().astype(np.float32, copy=False)         # Ergebnis zurück als NumPy-Array (auf CPU)

# -----------------------------
# SQLite PRAGMAs (Read-Only schnell)
# -----------------------------
def _apply_fast_pragmas(conn: sqlite3.Connection):
    try:
        conn.execute("PRAGMA journal_mode=WAL;")                   # WAL-Modus (Write-Ahead Logging) → bessere parallele Leseperformance
        conn.execute("PRAGMA synchronous=NORMAL;")                 # Sync-Level "NORMAL": schneller, etwas weniger robust als FULL
        conn.execute("PRAGMA temp_store=MEMORY;")                  # Temporäre Tabellen im RAM statt auf Festplatte
        conn.execute("PRAGMA cache_size=-400000;")                 # Setzt Cachegröße (~400MB im RAM, negatives Vorzeichen = KB)
        try: conn.execute("PRAGMA mmap_size=30000000000;")         # Memory-Mapped Files für schnelleren Zugriff (30GB Limit)
        except Exception: pass                                     # Falls System mmap nicht unterstützt → Fehler ignorieren
    except Exception:
        pass                                                       # Falls etwas nicht klappt → einfach weitermachen

# -----------------------------
# DB-Iterators
# -----------------------------
def _iter_image_id_pages(conn: sqlite3.Connection, page: int) -> Iterable[List[Tuple[int,str]]]:
    last_id = 0
    while True:
        rows = conn.execute(                                       # Hole eine Seite von Bild-IDs + Pfaden
            "SELECT image_id, path FROM images WHERE image_id > ? ORDER BY image_id LIMIT ?",
            (last_id, page)
        ).fetchall()
        if not rows: break                                         # Abbruchbedingung: keine neuen Zeilen mehr
        last_id = rows[-1][0]                                      # Merke die letzte ID, damit die nächste Abfrage dort weitermacht
        yield rows                                                 # Gibt die aktuelle Seite zurück

def stream_missing_embeddings_fs(conn: sqlite3.Connection, page: int,
                                 root: str, shard_mod: int, shard_width: int,
                                 limit: Optional[int]=None) -> Iterable[List[Tuple[int,str]]]:
    emitted = 0                                                    # Zähler für bereits "ausgegebene" fehlende Embeddings
    for rows in _iter_image_id_pages(conn, page):                  # Iteriere seitenweise über alle Bilder in DB
        todo = []
        for iid, p in rows:                                        # Für jede Bild-ID und ihren Pfad
            if not _exists_embed(root, int(iid), shard_mod, shard_width):  # Prüfen ob Embedding schon auf Festplatte existiert
                todo.append((int(iid), p))                         # Falls nicht vorhanden → in "todo"-Liste aufnehmen
                emitted += 1
                if limit is not None and emitted >= limit:         # Falls Obergrenze erreicht → abbrechen
                    break
        if todo:                                                   # Falls in dieser Seite fehlende Embeddings gefunden wurden
            yield todo                                             # Gibt Liste der fehlenden Einträge zurück
        if limit is not None and emitted >= limit:                 # Abbruch wenn Limit erreicht
            break


# ----------------------------- 
# Exact RAM Index (aus NPYs) 
# ----------------------------- 
class EmbedIndexExact: 
    def __init__(self, paths: List[str], mat: np.ndarray): 
        self.paths = paths                                                # Liste mit Bildpfaden (Mapping von Index → Bild)
        self.mat = mat.astype(np.float32, copy=False)                     # Alle Embeddings in einer großen Matrix (float32)

    def search(self, q: np.ndarray, k:int=5): 
        q = q.astype(np.float32); q /= (np.linalg.norm(q)+1e-8)           # Query-Vektor normalisieren (Länge = 1)
        k = min(k, len(self.paths))                                       # K darf nicht größer sein als die Anzahl der Bilder
        sims = self.mat @ q                                               # Skalarprodukt: Ähnlichkeit zwischen Query und allen Embeddings
        idx = np.argpartition(-sims, k-1)[:k]                             # Schnelle Auswahl der Top-k Indizes (ohne vollständiges Sortieren)
        idx = idx[np.argsort(-sims[idx])]                                 # Sortiert die k Kandidaten nach absteigender Ähnlichkeit
        return [(self.paths[int(i)], float(sims[int(i)])) for i in idx]   # Gibt Pfad + Score für die Top-k Ergebnisse zurück

def _load_all_embeds_from_npy(conn: sqlite3.Connection, outdir: Optional[str], 
                              shard_mod:int, shard_width:int, 
                              limit: Optional[int]=None) -> Tuple[List[str], np.ndarray]: 
    root = _feature_dir(conn.execute("PRAGMA database_list").fetchone()[2], outdir)   # Bestimmt Feature-Verzeichnis
    paths, vecs = [], [] 
    count = 0 
    cur = conn.cursor() 
    cur.execute("SELECT image_id, path FROM images ORDER BY image_id")                # Hole alle Bilder mit ID & Pfad aus DB
    while True: 
        rows = cur.fetchmany(100000)                                                 # Lade in Blöcken von 100k → Speicherfreundlich
        if not rows: break 
        for iid, p in rows: 
            npy = _npy_path(root, int(iid), shard_mod, shard_width)                  # Ermittelt NPY-Dateipfad für Bild
            if not os.path.exists(npy):                                              # Falls Embedding-Datei fehlt → überspringen
                continue 
            v = np.load(npy, mmap_mode="r").astype(np.float32, copy=False)           # Lade Embedding (Lazy Load via mmap)
            n = float(np.linalg.norm(v))                                             # Norm berechnen (Länge des Vektors)
            if n == 0 or not np.isfinite(n):                                         # Falls ungültig → überspringen
                continue 
            v = v / n                                                                # Normalisiere Embedding (Länge = 1)
            paths.append(p)                                                          # Speichere Bildpfad
            vecs.append(v)                                                           # Speichere normiertes Embedding
            count += 1 
            if limit is not None and count >= limit:                                 # Falls Limit erreicht → Abbruch
                break 
        if limit is not None and count >= limit: 
            break 
    if not vecs:                                                                     # Falls keine Embeddings geladen wurden
        return [], np.zeros((0,), dtype=np.float32) 
    return paths, np.vstack(vecs).astype(np.float32, copy=False)                     # Rückgabe: Liste von Pfaden + Embedding-Matrix

def load_exact_index(db_path:str, outdir: Optional[str]=None, 
                     shard_mod:int=DEFAULT_SHARD_MOD, shard_width:int=DEFAULT_SHARD_WIDTH) -> EmbedIndexExact: 
    conn = open_db(db_path); _apply_fast_pragmas(conn)                               # Öffne DB und setze Performance-PRAGMAs
    paths, M = _load_all_embeds_from_npy(conn, outdir, shard_mod, shard_width)       # Lade alle vorhandenen Embeddings aus NPY-Dateien
    conn.close() 
    return EmbedIndexExact(paths=paths, mat=M)                                       # Erstelle Index-Objekt mit Embeddings im RAM


# -----------------------------
# HNSW – Build/Load/Search
# -----------------------------
import hnswlib

def _hnsw_base(db_path:str, outdir: Optional[str]=None) -> str:
    base = os.path.abspath(outdir) if outdir else os.path.abspath(os.path.dirname(db_path) or ".") # Basisverzeichnis bestimmen
    return os.path.join(base, "features", "embed", "hnsw")                                         # Standard-Speicherort für HNSW-Index

def build_hnsw_index(db_path:str, dim:int=1280, M:int=32, ef_construction:int=200,
                     save:bool=True, outdir: Optional[str]=None,
                     shard_mod:int=DEFAULT_SHARD_MOD, shard_width:int=DEFAULT_SHARD_WIDTH,
                     limit: Optional[int]=None):
    conn = open_db(db_path); _apply_fast_pragmas(conn)                                              # DB öffnen + Performance-Optimierungen aktivieren
    ids, vecs = [], []
    root = _feature_dir(db_path, outdir)                                                            # Pfad zum Embedding-Verzeichnis
    cur = conn.cursor(); cur.execute("SELECT image_id FROM images ORDER BY image_id")               # Alle Bild-IDs holen
    seen = 0
    with tqdm(desc="HNSW Build: load NPY", unit="img", dynamic_ncols=True) as pbar:                 # Fortschrittsanzeige
        while True:
            rows = cur.fetchmany(100000)                                                            # Lade Bild-IDs seitenweise
            if not rows: break
            for (iid,) in rows:
                npy = _npy_path(root, int(iid), shard_mod, shard_width)                             # Pfad zur NPY-Datei für das Bild
                if not os.path.exists(npy):                                                         # Falls keine Embedding-Datei existiert → überspringen
                    continue
                v = np.load(npy, mmap_mode="r").astype(np.float32, copy=False)                      # Embedding laden (Lazy Load via mmap)
                n = float(np.linalg.norm(v))                                                        # Norm des Vektors berechnen
                if n == 0 or not np.isfinite(n):                                                    # Falls ungültig → überspringen
                    continue
                v = v / n                                                                           # Normalisierung auf Länge = 1
                ids.append(int(iid))                                                                # ID speichern
                vecs.append(v)                                                                      # Embedding speichern
                seen += 1
                pbar.update(1)                                                                      # Fortschritt aktualisieren
                if limit is not None and seen >= limit:                                             # Falls Limit erreicht → Abbruch
                    break
            if limit is not None and seen >= limit:
                break
    conn.close()
    if not ids:                                                                                     # Falls keine Embeddings geladen → Fehler
        raise ValueError("Keine Embeddings (NPY) gefunden – bitte erst --stage embed ausführen.")

    ids = np.asarray(ids, dtype=np.int64)                                                           # IDs in NumPy-Array
    data = np.vstack(vecs).astype(np.float32, copy=False)                                           # Embeddings in große Matrix stapeln

    index = hnswlib.Index(space='cosine', dim=dim)                                                  # Neuen HNSW-Index anlegen (Cosine-Similarity)
    index.init_index(max_elements=data.shape[0], M=M, ef_construction=ef_construction, random_seed=42)  # Initialisieren mit Parametern
    index.add_items(data, ids)                                                                      # Embeddings + IDs in Index einfügen
    index.set_ef(100)                                                                               # Such-Parameter (Trade-off Speed/Genauigkeit)

    if save:                                                                                        # Falls Speichern aktiviert
        base = _hnsw_base(db_path, outdir)                                                          # Speicherpfad bestimmen
        os.makedirs(os.path.dirname(base), exist_ok=True)                                           # Ordner anlegen
        index.save_index(base + ".bin")                                                             # Index in Binärdatei speichern
        np.save(base + ".ids.npy", ids)                                                             # IDs separat abspeichern
    return index

def load_hnsw_index(db_path:str, dim:int=1280, ef:int=100, outdir: Optional[str]=None):
    base = _hnsw_base(db_path, outdir)                                                              # Speicherpfad bestimmen
    index = hnswlib.Index(space='cosine', dim=dim)                                                  # HNSW-Index anlegen
    index.load_index(base + ".bin")                                                                 # Gespeicherten Index laden
    index.set_ef(ef)                                                                                # Suchparameter einstellen
    return index


# ----------------------------- 
# Query-Vektor 
# ----------------------------- 
@torch.inference_mode()                                                   # Schaltet Gradienten aus → schneller/speichersparend
def _compute_query_vec(path: str) -> Optional[np.ndarray]: 
    t = _preprocess(path)                                                 # Bild laden + normalisieren + in Tensor umwandeln
    if t is None: return None                                             # Abbruch, falls Bild nicht lesbar
    t = t.unsqueeze(0)                                                    # Batch-Dimension hinzufügen (1, C, H, W)
    if AMP:                                                               # Falls Automatic Mixed Precision aktiviert
        with torch.autocast(device_type=_DEVICE.type, dtype=torch.float16):  # Rechne in float16 für Effizienz (auf GPU sinnvoll)
            feat = _BACKBONE(t.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True) if _DEVICE.type == "cuda" else t)  # Forward-Pass
    else: 
        feat = _BACKBONE(t.to(device=_DEVICE, dtype=_TORCH_DTYPE, non_blocking=True) if _DEVICE.type == "cuda" else t)       # Normaler Forward-Pass
    v = feat.float().squeeze(0).cpu().numpy().astype(np.float32)          # Zurück zu float32, Batch weg, auf CPU, NumPy-Array
    v /= (np.linalg.norm(v)+1e-8)                                         # L2-Normierung (Cosine-kompatibel, stabilisiert durch 1e-8)
    return v                                                              # Rückgabe: normalisierter Query-Vektor
 
# ----------------------------- 
# HNSW-Suche 
# ----------------------------- 
def search_hnsw(db_path:str, qvec: Optional[np.ndarray]=None, k:int=5, 
                query_path: Optional[str]=None, outdir: Optional[str]=None) -> List[Tuple[float,int,str]]: 
    if qvec is None:                                                      # Wenn kein Query-Vektor gegeben ist …
        if not query_path: 
            raise ValueError("Entweder qvec oder query_path angeben.")    # … dann muss ein Bildpfad da sein
        qvec = _compute_query_vec(query_path)                             # Query-Vektor aus Bild berechnen
        if qvec is None: 
            return []                                                     # Nichts gefunden, wenn Bild nicht lesbar
    idx = load_hnsw_index(db_path, dim=len(qvec), ef=100, outdir=outdir)  # HNSW-Index laden (Cosine, ef=100 für ordentliche Qualität)
    labels, dists = idx.knn_query(qvec.reshape(1,-1).astype(np.float32), k=k)  # k nächste Nachbarn zum Query suchen
    labels, dists = labels[0], dists[0]                                   # Ergebnisse aus Batch-Form entpacken
    conn = open_db(db_path)                                               # DB öffnen, um Pfade zu den IDs nachzuschlagen
    out: List[Tuple[float,int,str]] = [] 
    for lab, dist in zip(labels, dists):                                  # IDs und Distanzen zusammen durchgehen
        row = conn.execute("SELECT path FROM images WHERE image_id=?", (int(lab),)).fetchone()  # Pfad für Bild-ID holen
        if row: 
            out.append((float(dist), int(lab), row[0]))                   # (Distanz, ID, Pfad) anhängen
    conn.close() 
    out.sort(key=lambda x: x[0])                                          # Sicherheitshalber nach Distanz sortieren (kleiner = ähnlicher)
    return out                                                            # Rückgabe der Trefferliste

# -----------------------------
# EMBEDDING-PIPELINE (hoch performant, 1 Progressbar)
# -----------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed        # Für parallele I/O-Aufgaben (Bilder laden etc.)
IO_WORKERS   = int(os.environ.get("IMG_IO_WORKERS", "24"))             # Anzahl Threads für I/O (über ENV variierbar)
QUEUE_MAX    = int(os.environ.get("IMG_QUEUE_MAX", "2048"))            # Max. Queue-Größe für Pipeline-Puffer
FLUSH_SECS   = float(os.environ.get("IMG_FLUSH_SECS", "0.5"))          # Flush-Intervall (Sekunden) für Batch-Schreiben

@torch.inference_mode()                                                # Keine Gradienten → weniger Overhead
def process_embeddings(conn: sqlite3.Connection, mini_batch:int=384, limit: Optional[int]=None,
                       outdir: Optional[str]=None, shard_mod:int=DEFAULT_SHARD_MOD,
                       shard_width:int=DEFAULT_SHARD_WIDTH, dtype: str="fp16") -> None:

                                                                       # Nur Dateisystem-Output, keine DB-Writes
    import queue, threading                                            # Lokale Imports für Threading/Queues

    global SAVE_DTYPE                                                  # Globalen Speichertyp setzen (für NPY)
    SAVE_DTYPE = np.float16 if dtype == "fp16" else np.float32         # fp16 spart Speicher/IO, fp32 für Präzision

    create_schema(conn)                                                # Stellt sicher, dass DB-Schema existiert
    _apply_fast_pragmas(conn)                                          # SQLite auf schnelle Lese-/Cache-Settings drehen

    # Zielverzeichnis (Standard: neben DB)
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]       # Pfad zur geöffneten DB ermitteln
    root = _feature_dir(db_path, outdir)                               # Basisordner für feature/embed/* anlegen

    # --- Pre-Scan: wie viele fehlen? (für saubere Progressbar)
    missing_ids = []                                                   # Liste der (image_id, path), die noch fehlen
    cur = conn.cursor()                                                # Cursor für seitenweises Lesen
    cur.execute("SELECT image_id, path FROM images ORDER BY image_id") # Alle Bilder geordnet abfragen
    scanned = 0                                                        # Zähler, nur für Info/Debug
    while True:
        rows = cur.fetchmany(200000)                                   # Große Seiten (200k) → wenige DB-Roundtrips
        if not rows: break                                             # Abbruch, wenn nichts mehr kommt
        for iid, p in rows:                                            # Über alle Einträge der Seite iterieren
            if not _exists_embed(root, int(iid), shard_mod, shard_width):  # Prüfen, ob NPY für dieses Bild fehlt
                missing_ids.append((int(iid), p))                      # Fehlt → zur To-Do-Liste hinzufügen
                if limit is not None and len(missing_ids) >= limit:    # Optionales Oberlimit beachten
                    break
        if limit is not None and len(missing_ids) >= limit:            # Seite verlassen, wenn Limit erreicht
            break
        scanned += len(rows)                                           # Fortschritt (nur metrisch, hier nicht genutzt)

    total_missing = len(missing_ids)                                   # Anzahl der tatsächlich fehlenden Embeddings
    if total_missing == 0:                                             # Nichts zu tun?
        print("Alles erledigt: keine fehlenden Embeddings gefunden.")  # Kurze Erfolgsmeldung
        return                                                         # Frühzeitiger Exit

    q: "queue.Queue[Tuple[int, Optional[torch.Tensor]]]" = queue.Queue(maxsize=QUEUE_MAX) 
    stop_token = ( -1, None )                                               # Spezielles Token zum Stoppen des Consumers
 
    def producer(): 
        # Parallel: I/O + Preprocess 
        def work(item):                                                     # Worker-Funktion für einzelne Items
            iid, path = item                                                # Bild-ID und Pfad entpacken
            t = _preprocess(path)                                           # Bild laden + preprocessen
            return iid, t                                                   # Rückgabe: ID + Tensor (oder None, falls Fehler)
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:              # ThreadPool für paralleles I/O + Preprocessing
            for iid, t in ex.map(work, missing_ids, chunksize=64):          # Mappt Items → verarbeitet in Blöcken von 64
                q.put((iid, t))                                             # Ergebnisse in Queue einfügen
        q.put(stop_token)                                                   # Signal an Consumer: fertig
 
    prod_th = threading.Thread(target=producer, daemon=True)                # Producer-Thread starten (läuft im Hintergrund)
    prod_th.start() 
 
    ok = fail = 0                                                           # Zähler: erfolgreiche/fehlgeschlagene Embeddings
    last_flush = time.time()                                                # Zeitpunkt des letzten Flushs
    batch_ids: List[int] = []                                               # IDs für den aktuellen Mini-Batch
    batch_tensors: List[torch.Tensor] = []                                  # Tensoren für den aktuellen Mini-Batch
 
    # TQDM: 1 Leiste über alle fehlenden 
    with tqdm(total=total_missing, desc="Embeddings", unit="img", 
              dynamic_ncols=True, smoothing=0.1) as pbar:                   # Fortschrittsbalken für alle fehlenden Embeddings
 
        cur_batch = mini_batch                                              # Startgröße für Mini-Batches
        while True: 
            iid, t = q.get()                                                # Hole nächstes Item aus Queue
            if (iid, t) == stop_token:                                      # Falls Stop-Signal → Consumer beenden
                # Rest flushen 
                if batch_tensors:                                           # Falls noch unverarbeitete Tensoren im Batch sind
                    try: 
                        feats = _infer_batch(batch_tensors)                 # Forward-Pass durchs Modell
                        for i, bi in enumerate(batch_ids):                  # Ergebnisse speichern
                            _save_npy_atomic(_npy_path(root, bi, shard_mod, shard_width), feats[i]) 
                            ok += 1                                         # Erfolgreich gespeichert
                        pbar.update(len(batch_ids))                         # Fortschrittsbalken updaten
                    except RuntimeError as e: 
                        # OOM Fallback für den letzten Rest 
                        if "CUDA" in str(e).upper() and len(batch_tensors) > 1:  # Falls Out-of-Memory auf GPU
                            cur_batch = max(1, len(batch_tensors)//2)       # Batch halbieren und erneut probieren
                            # split & retry 
                            for s in range(0, len(batch_tensors), cur_batch): 
                                sub_t = batch_tensors[s:s+cur_batch]        # Teilbatch Tensoren
                                sub_i = batch_ids[s:s+cur_batch]            # Teilbatch IDs
                                feats = _infer_batch(sub_t)                 # Nochmal Inferenz für den Teilbatch
                                for i, bi in enumerate(sub_i):              # Ergebnisse speichern
                                    _save_npy_atomic(_npy_path(root, bi, shard_mod, shard_width), feats[i]) 
                                    ok += 1 
                                pbar.update(len(sub_i))                     # Fortschrittsbalken für Teilbatch
                        else: 
                            fail += len(batch_tensors)                      # Falls anderer Fehler → alles als Fail zählen
                break                                                       # Abbruch der While-Schleife

            if t is None:                                               # Falls Preprocessing für Bild fehlschlug
                fail += 1                                               # Fehler-Zähler hochzählen
                pbar.update(1)                                          # Fortschrittsbalken trotzdem fortschreiben
                continue                                                # Dieses Bild überspringen

            batch_ids.append(iid)                                       # ID in aktuelle Batch-Liste einfügen
            batch_tensors.append(t)                                     # Tensor in Batch aufnehmen

            # Zeitbasiertes Flush (verhindert Idle bei langsamer Producer-Rate)
            need_time_flush = (time.time() - last_flush) >= FLUSH_SECS  # Falls zu lange seit letztem Flush → sofort verarbeiten

            if len(batch_tensors) >= cur_batch or need_time_flush:      # Batch voll ODER Zeitlimit erreicht
                try:
                    feats = _infer_batch(batch_tensors)                 # Embeddings im Batch berechnen
                    for i, bi in enumerate(batch_ids):                  # Ergebnisse einzeln speichern
                        _save_npy_atomic(_npy_path(root, bi, shard_mod, shard_width), feats[i])
                        ok += 1                                         # Erfolgreich geschrieben
                    pbar.update(len(batch_ids))                         # Fortschrittsbalken updaten
                    # Durchsatz anzeigen
                    pbar.set_postfix_str(f"batch={cur_batch}, ok={ok}, fail={fail}")  # Status-Info im TQDM-Balken
                    batch_ids.clear()                                   # Batch-Listen leeren
                    batch_tensors.clear()
                    last_flush = time.time()                            # Zeitstempel aktualisieren
                    # nach Erfolg: Batch wieder anheben (sanft)
                    if cur_batch < mini_batch:                          # Falls Batch zuvor verkleinert wurde
                        cur_batch = min(mini_batch, max(1, cur_batch*2))# Verdoppeln (bis max mini_batch) für mehr Durchsatz
                except RuntimeError as e:
                    # CUDA OOM -> Batch halbieren und sofort neu versuchen
                    if "CUDA" in str(e).upper() and cur_batch > 1:      # Falls Out-Of-Memory
                        cur_batch = max(1, cur_batch // 2)              # Batch-Größe halbieren
                        # nichts schreiben, direkt mit kleinerem Batch weitermachen
                    else:
                        # irreparabler Fehler -> diese Items zählen als fail
                        fail += len(batch_tensors)                      # Alle Bilder im Batch als Fehler markieren
                        pbar.update(len(batch_tensors))                 # Fortschrittsbalken fortschreiben
                        batch_ids.clear()                               # Batch leeren
                        batch_tensors.clear()
                        last_flush = time.time()                        # Zeitstempel aktualisieren

    print(f"Embeddings (NPY) fertig: {ok} ok, {fail} Fehler")           # Zusammenfassung: wie viele erfolgreich/fehlgeschlagen

# ----------------------------- 
# CLI 
# ----------------------------- 
if __name__ == "__main__":                                               # Nur ausführen, wenn Script direkt gestartet wird
    import argparse 
    parser = argparse.ArgumentParser()                                   # Argumentparser für Kommandozeile
    parser.add_argument("--stage",                                       # Welche Hauptoperation soll ausgeführt werden?
        choices=["embed", "build_index_hnsw", "search_hnsw", "search_exact"], 
        required=True)                                                   # Muss angegeben werden
    parser.add_argument("--db", default=cfg.DB_PATH)                     # Pfad zur SQLite-Datenbank
    parser.add_argument("--query", type=str)                             # Pfad zum Query-Bild (nur für Suche)
    parser.add_argument("--k", type=int, default=5)                      # Anzahl Ergebnisse bei der Suche
    parser.add_argument("--batch", type=int, default=384)                # Batch-Größe für Embedding-Inferenz
    parser.add_argument("--limit", type=int, default=None)               # Optionales Limit (z. B. max. Anzahl Embeddings)
    parser.add_argument("--outdir", type=str, default=None)              # Zielordner (falls anders als Standard)
    parser.add_argument("--shard_mod", type=int, default=DEFAULT_SHARD_MOD)     # Sharding-Modulus
    parser.add_argument("--shard_width", type=int, default=DEFAULT_SHARD_WIDTH) # Shard-Verzeichnisbreite (Ziffern)
    parser.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")     # Speichertyp für Embeddings (Platz vs. Genauigkeit)
    args = parser.parse_args()                                           # Argumente parsen
 
    if args.stage == "embed":                                            # Stage: Embeddings berechnen und speichern
        conn = open_db(args.db, bulk=True)                               # DB öffnen im Bulk-Modus
        create_schema(conn)                                              # DB-Schema anlegen (falls fehlt)
        _apply_fast_pragmas(conn)                                        # Performance-PRAGMAs setzen
        process_embeddings(                                              # Embeddings verarbeiten und abspeichern
            conn, mini_batch=args.batch, limit=args.limit, 
            outdir=args.outdir, shard_mod=args.shard_mod, 
            shard_width=args.shard_width, dtype=args.dtype 
        ) 
        conn.close()                                                     # DB schließen
 
    elif args.stage == "build_index_hnsw":                               # Stage: HNSW-Index bauen
        build_hnsw_index( 
            args.db, dim=1280, M=32, ef_construction=200, save=True, 
            outdir=args.outdir, shard_mod=args.shard_mod, 
            shard_width=args.shard_width, limit=args.limit 
        ) 
        print("HNSW-Index built and saved.")                             # Rückmeldung an Nutzer
 
    elif args.stage == "search_hnsw":                                    # Stage: Suche über HNSW-Index
        if not args.query: 
            print("Bitte --query angeben."); exit(1)                     # Query-Bild muss angegeben sein
        q = _compute_query_vec(args.query)                               # Query-Vektor berechnen
        if q is None: 
            print("Query-Bild konnte nicht verarbeitet werden."); exit(1)# Falls Laden fehlgeschlagen
        hits = search_hnsw(args.db, qvec=q, k=args.k, outdir=args.outdir)# HNSW-Suche durchführen
        for dist, iid, p in hits:                                        # Treffer ausgeben
            score = 1.0 - dist                                           # Cosine-Similarity = 1 - Distanz
            print(f"{score:.4f}\t{iid}\t{p}")                            # Score, ID und Bildpfad
 
    elif args.stage == "search_exact":                                   # Stage: Exakte Suche (brute force)
        if not args.query: 
            print("Bitte --query angeben."); exit(1)                     # Query-Bild muss angegeben sein
        q = _compute_query_vec(args.query)                               # Query-Vektor berechnen
        if q is None: 
            print("Query-Bild konnte nicht verarbeitet werden."); exit(1)
        idx = load_exact_index(args.db, outdir=args.outdir, shard_mod=args.shard_mod, shard_width=args.shard_width) 
        hits = idx.search(q, k=args.k)  # [(path, score)]                # Exakte Suche mit RAM-Index
        for p, s in hits:                                                # Treffer ausgeben
            print(f"{s:.4f}\t{p}")                                       # Score und Pfad