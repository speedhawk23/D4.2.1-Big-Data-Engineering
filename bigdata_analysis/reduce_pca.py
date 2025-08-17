# reduce_pca.py
import argparse, glob, numpy as np
from sklearn.decomposition import IncrementalPCA

def load_vec(path):
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    else:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="features/embed/**/*.npy",
                    help="Pattern der einzelnen Embedding-Dateien")
    ap.add_argument("--out", default="embeds_pca64.npy",
                    help="Ausgabedatei (N,64) – NumPy memmap")
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob, recursive=True))
    if not files:
        raise SystemExit(f"Keine Dateien gefunden (Pattern: {args.in_glob}).")

    # 1) Erste gültige Datei finden -> D bestimmen
    D = None
    for f in files:
        v = load_vec(f)
        if v is not None:
            D = v.shape[0]
            break
    if D is None:
        raise SystemExit("Keine passenden 1D-Embeddings gefunden.")

    # 2) Nur gültige Dateien sammeln (1D, Länge == D)
    good, bad = [], []
    for f in files:
        try:
            v = load_vec(f)
            if v is not None and v.shape[0] == D:
                good.append(f)
            else:
                # shape merken (z.B. (50000,))
                arr = np.load(f, mmap_mode="r", allow_pickle=False)
                bad.append((f, tuple(arr.shape)))
        except Exception as e:
            bad.append((f, str(e)))

    N = len(good)
    if N == 0:
        raise SystemExit(f"Kein passendes Embedding gefunden. Beispiele geskippt: {bad[:3]}")
    print(f"Gefunden: {N} gültige Embeddings (D={D}). Geskippt: {len(bad)}")
    if bad:
        print("Beispiele geskippt:", bad[:3])

    # 3) Incremental PCA: partial_fit
    ipca = IncrementalPCA(n_components=args.dims)
    for s in range(0, N, args.batch):
        batch_files = good[s:s+args.batch]
        X = np.stack([np.asarray(load_vec(f), dtype="float32") for f in batch_files])
        ipca.partial_fit(X)
        print(f"partial_fit: {min(s+len(batch_files), N)}/{N}")

    # 4) Transform + speichern (memmap .npy)
    X_red = np.memmap(args.out, dtype="float32", mode="w+",
                      shape=(N, args.dims))
    row = 0
    for s in range(0, N, args.batch):
        batch_files = good[s:s+args.batch]
        X = np.stack([np.asarray(load_vec(f), dtype="float32") for f in batch_files])
        X64 = ipca.transform(X)
        X_red[row:row+X64.shape[0]] = X64
        row += X64.shape[0]
        print(f"transform: {row}/{N}")
    del X_red  # flush
    print(f"PCA fertig → {args.out}  Shape=({N},{args.dims})")

if __name__ == "__main__":
    main()
