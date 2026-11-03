"""
MallSegments - Customer Segmentation with Clustering

Columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

Usage:  python clustering_project.py data.csv
If no data file is given, a synthetic dataset with the same structure is
generated (only for testing the code).
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN,
                             MeanShift, estimate_bandwidth, Birch,
                             SpectralClustering, AffinityPropagation, OPTICS)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
RS = 42


# ---------- 1. Loading and cleaning ----------
def load(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        rng = np.random.default_rng(RS)
        centers = [(25, 25, 78), (45, 26, 20), (32, 80, 82), (43, 87, 18), (45, 55, 50)]
        rows = []
        for a, i, s in centers:
            n = 40
            rows.append(np.c_[rng.normal(a, 6, n), rng.normal(i, 8, n), rng.normal(s, 8, n)])
        X = np.vstack(rows).clip(18, 100)
        df = pd.DataFrame(X, columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
        df.insert(0, "Gender", rng.choice(["Male", "Female"], len(df)))
        df.insert(0, "CustomerID", range(1, len(df) + 1))
    # Normalize column names
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "income" in lc: ren[c] = "Income"
        elif "spending" in lc: ren[c] = "Score"
        elif lc == "age": ren[c] = "Age"
        elif "gender" in lc or "genre" in lc: ren[c] = "Gender"
        elif "id" in lc: ren[c] = "CustomerID"
    df = df.rename(columns=ren).drop_duplicates().dropna()
    return df


df = load(sys.argv[1] if len(sys.argv) > 1 else None)
print(df.describe(include="all"))

# ---------- 2. Feature engineering ----------
df["GenderCode"] = (df["Gender"].str.lower().str.startswith("m")).astype(int)
FEATURE_SETS = {
    "Income+Score": ["Income", "Score"],
    "Age+Income+Score": ["Age", "Income", "Score"],
}

# ---------- 3. Choosing K (Elbow + Silhouette) ----------
def k_selection(X, name):
    ks = range(2, 11)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RS).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("K")
    ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("K")
    plt.tight_layout(); plt.savefig(f"k_selection_{name}.png", dpi=120); plt.close()
    return list(ks)[int(np.argmax(sil))]


# ---------- 4. Evaluation ----------
def evaluate(X, labels):
    mask = labels != -1            # DBSCAN/OPTICS noise points are excluded
    n = len(set(labels[mask]))
    if n < 2 or mask.sum() <= n:
        return None
    return dict(n_clusters=n, noise=int((~mask).sum()),
                silhouette=silhouette_score(X[mask], labels[mask]),
                calinski=calinski_harabasz_score(X[mask], labels[mask]),
                davies_bouldin=davies_bouldin_score(X[mask], labels[mask]))


def run_all(X, k):
    bw = estimate_bandwidth(X, quantile=0.2, random_state=RS)
    models = {
        "KMeans": KMeans(k, n_init=10, random_state=RS),
        "Agglomerative-ward": AgglomerativeClustering(k, linkage="ward"),
        "Agglomerative-average": AgglomerativeClustering(k, linkage="average"),
        "GMM": GaussianMixture(k, random_state=RS, n_init=5),
        "Birch": Birch(n_clusters=k),
        "Spectral": SpectralClustering(k, affinity="nearest_neighbors", random_state=RS),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "OPTICS": OPTICS(min_samples=8, xi=0.05),
        "MeanShift": MeanShift(bandwidth=bw if bw > 0 else None),
        "AffinityPropagation": AffinityPropagation(random_state=RS, damping=0.8),
    }
    res, labs = {}, {}
    for name, m in models.items():
        try:
            lab = m.fit_predict(X)
            r = evaluate(X, lab)
            if r:
                res[name] = r; labs[name] = lab
        except Exception as e:
            print(f"{name} failed: {e}")
    return pd.DataFrame(res).T.sort_values("silhouette", ascending=False), labs


# ---------- 5. Run on each feature set ----------
summary = []
best = None
for fs_name, cols in FEATURE_SETS.items():
    X = StandardScaler().fit_transform(df[cols])
    k = k_selection(X, fs_name)
    table, labs = run_all(X, k)
    print(f"\n=== {fs_name} (best K by silhouette = {k}) ===")
    print(table.round(3))
    table.insert(0, "features", fs_name)
    summary.append(table)
    # Only models that label at most 5% of the data as noise may be selected
    ok = table[table["noise"] <= 0.05 * len(df)]
    top = ok.index[0]
    if best is None or ok.iloc[0]["silhouette"] > best[0]:
        best = (ok.iloc[0]["silhouette"], fs_name, top, labs[top], cols, X)

pd.concat(summary).to_csv("model_comparison.csv")

# ---------- 6. Best model: cluster profiles ----------
_, fs_name, model_name, labels, cols, X = best
df["Cluster"] = labels
print(f"\nBest: {model_name} on {fs_name}")
profile = df[df.Cluster != -1].groupby("Cluster").agg(
    Count=("CustomerID", "count"), Age=("Age", "mean"), Income=("Income", "mean"),
    Score=("Score", "mean"), MaleRatio=("GenderCode", "mean")).round(1)
print(profile)
profile.to_csv("cluster_profiles.csv")
df.to_csv("customers_with_clusters.csv", index=False)

# Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(df.Income, df.Score, c=labels, cmap="tab10", s=40)
ax[0].set_xlabel("Annual Income"); ax[0].set_ylabel("Spending Score")
ax[0].set_title(f"{model_name} - Income vs Score")
p = PCA(2, random_state=RS).fit_transform(X)
ax[1].scatter(p[:, 0], p[:, 1], c=labels, cmap="tab10", s=40)
ax[1].set_title("PCA projection")
plt.tight_layout(); plt.savefig("clusters.png", dpi=120); plt.close()
print("Saved: model_comparison.csv, cluster_profiles.csv, customers_with_clusters.csv, clusters.png")
