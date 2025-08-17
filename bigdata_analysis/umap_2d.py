import argparse, numpy as np, pandas as pd
import umap, plotly.express as px

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="embeds_pca64.npy")
    ap.add_argument("--out_html", default="umap_2d.html")
    ap.add_argument("--out_csv", default="umap_2d_coords.csv")
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.05)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = np.load(args.inp)  # (N,64)
    print("Geladen:", args.inp, X.shape)

    um = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed
    )
    Y = um.fit_transform(X)  # (N,2)
    ids = np.arange(Y.shape[0])

    df = pd.DataFrame({"x": Y[:,0], "y": Y[:,1], "id": ids})
    df.to_csv(args.out_csv, index=False)

    fig = px.scatter(df, x="x", y="y", hover_data=["id"], render_mode="webgl", height=900)
    fig.write_html(args.out_html, include_plotlyjs="cdn")

    print("Gespeichert:", args.out_html, "und", args.out_csv)

if __name__ == "__main__":
    main()
