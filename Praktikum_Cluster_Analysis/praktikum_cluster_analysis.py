# -*- coding: utf-8 -*-
# %% [markdown]
# # Praktikum Cluster Analysis — K-Means, Hierarchical, dan DBSCAN pada Data Besar
# Pembelajaran Mesin — Informatika, Universitas Sanata Dharma
#
# Setelah mempelajari K-Means, Hierarchical Clustering, dan DBSCAN satu per satu,
# praktikum ini menerapkan KETIGANYA pada dataset yang sama. Tujuannya agar
# mahasiswa memahami bahwa cluster analysis bukan sekadar menjalankan algoritma.
# Prosesnya mencakup:
#   1. memahami data dan menyiapkannya (cleaning, scaling, mengatasi fitur redundan),
#   2. memilih algoritma dan parameternya dengan alasan yang jelas,
#   3. mengevaluasi hasil secara internal (silhouette, DBI) dan, bila ada label,
#      secara eksternal (ARI, tabel silang),
#   4. menafsirkan setiap klaster: apa ciri khasnya?
#
# Dataset (pilih lewat variabel DATASET):
#   "bean"   → dataset_dry_bean.csv     13.611 biji kacang × 16 fitur bentuk.
#              Data ini punya label varietas (7 kelas). Label DISEMBUNYIKAN saat
#              clustering, lalu dipakai untuk mengecek apakah klaster yang
#              ditemukan sesuai dengan varietas sebenarnya. (DATASET UTAMA)
#   "kredit" → dataset_kartu_kredit.csv 8.950 nasabah × 17 fitur perilaku kartu
#              kredit, tanpa label. Ini kasus segmentasi nasabah yang nyata.
#              (TANTANGAN)
#
# Kebutuhan: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy.
# Hierarchical clustering pada seluruh data butuh RAM ±2 GB. Jika laptop Anda
# terbatas, set HIER_SAMPLE = 3000 (lihat bagian 6). Google Colab aman.

# %%
# =============================================================================
# 0. IMPORT DAN KONFIGURASI
# =============================================================================
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATASET = "bean"          # ganti ke "kredit" untuk dataset tantangan
RANDOM_STATE = 42
SIL_SAMPLE = 3000         # silhouette O(n^2): dihitung pada sampel agar cepat
OUT_DIR = f"output_{DATASET}"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.RandomState(RANDOM_STATE)

KONFIG = {
    "bean": dict(file="dataset_dry_bean.csv", label="Class", id_col=None,
                 log_cols=[]),
    "kredit": dict(file="dataset_kartu_kredit.csv", label=None, id_col="CUST_ID",
                   # kolom nominal uang dan jumlah transaksi sangat menceng → log(1+x)
                   log_cols=["BALANCE", "PURCHASES", "ONEOFF_PURCHASES",
                             "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "CASH_ADVANCE_TRX",
                             "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS",
                             "MINIMUM_PAYMENTS"]),
}
CFG = KONFIG[DATASET]


def simpan(fig, nama):
    """Simpan gambar ke folder output lalu tampilkan."""
    fig.savefig(os.path.join(OUT_DIR, nama), dpi=150, bbox_inches="tight")
    plt.show()


def sil(X, label):
    """Silhouette pada sampel acak; titik noise DBSCAN (-1) tidak diikutkan."""
    label = np.asarray(label)
    m = label != -1
    if len(np.unique(label[m])) < 2:
        return np.nan
    idx = np.where(m)[0]
    if len(idx) > SIL_SAMPLE:
        idx = rng.choice(idx, SIL_SAMPLE, replace=False)
    return silhouette_score(X[idx], label[idx])


# %% [markdown]
# ## 1. Memuat dan memahami data

# %%
df = pd.read_csv(CFG["file"])
print("Ukuran data awal:", df.shape)
print(df.head())
print("\nTipe data:\n", df.dtypes.value_counts())
if CFG["label"]:
    print("\nDistribusi label (HANYA untuk validasi, tidak dipakai saat clustering):")
    print(df[CFG["label"]].value_counts())

# %% [markdown]
# ## 2. Cleaning

# %%
print("Missing value per kolom:\n", df.isna().sum()[df.isna().sum() > 0])
print("Jumlah baris duplikat:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

if DATASET == "kredit":
    # 1 nasabah tanpa CREDIT_LIMIT → dibuang; MINIMUM_PAYMENTS kosong → median
    df = df.dropna(subset=["CREDIT_LIMIT"])
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())
    # frekuensi seharusnya 0..1, tetapi ada nilai 1,5 → kesalahan data, dipotong
    print("CASH_ADVANCE_FREQUENCY > 1:", (df["CASH_ADVANCE_FREQUENCY"] > 1).sum(), "baris")
    df["CASH_ADVANCE_FREQUENCY"] = df["CASH_ADVANCE_FREQUENCY"].clip(upper=1)
    df = df.reset_index(drop=True)

y = df[CFG["label"]] if CFG["label"] else None
FITUR = [c for c in df.columns if c not in (CFG["label"], CFG["id_col"])]
print(f"\nUkuran setelah cleaning: {df.shape} | jumlah fitur: {len(FITUR)}")

# %% [markdown]
# ## 3. Eksplorasi: distribusi dan korelasi

# %%
n = len(FITUR)
ncol = 6
fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(18, 2.6 * np.ceil(n / ncol)))
for a, f in zip(axes.ravel(), FITUR):
    sns.histplot(df[f], ax=a, bins=40, color="#5b9bd5")
    a.set_title(f, fontsize=8)
    a.set_xlabel("")
    a.set_ylabel("")
for a in axes.ravel()[n:]:
    a.axis("off")
fig.tight_layout()
simpan(fig, "01_distribusi_fitur.png")

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df[FITUR].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".1f",
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Korelasi antarfitur")
simpan(fig, "02_korelasi.png")

kor = df[FITUR].corr().abs().where(np.triu(np.ones((n, n)), 1).astype(bool)).stack()
print("Pasangan fitur dengan |korelasi| > 0,9 :", int((kor > 0.9).sum()))
print(kor[kor > 0.9].sort_values(ascending=False).head(10).round(3))

# %% [markdown]
# ## 4. Transformasi, standarisasi, dan PCA
# Banyak fitur saling berkorelasi tinggi. Jika semua dipakai mentah, informasi
# yang sama "terhitung berkali-kali" dalam jarak Euclidean. PCA membantu kita
# (a) melihat berapa dimensi yang sebenarnya penting dan (b) memvisualisasikan
# data dalam 2D.

# %%
X_df = df[FITUR].copy()
for c in CFG["log_cols"]:
    X_df[c] = np.log1p(X_df[c])
X = StandardScaler().fit_transform(X_df)

pca_full = PCA(random_state=RANDOM_STATE).fit(X)
kum = np.cumsum(pca_full.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(range(1, n + 1), pca_full.explained_variance_ratio_, color="#9dc3e6", label="per komponen")
ax.plot(range(1, n + 1), kum, "o-", color="#c00000", label="kumulatif")
ax.axhline(0.9, ls="--", color="grey")
ax.set(xlabel="Komponen utama", ylabel="Proporsi varians", title="Scree plot PCA")
ax.legend()
simpan(fig, "03_pca_scree.png")
print("Jumlah komponen untuk 90% varians:", int(np.argmax(kum >= 0.9) + 1))

P2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)   # untuk visualisasi

if y is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=P2[:, 0], y=P2[:, 1], hue=y, s=6, linewidth=0, palette="tab10", ax=ax)
    ax.set(title="Label asli pada bidang PCA (bukan input clustering)", xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=3, fontsize=8)
    simpan(fig, "04_label_asli_pca.png")

# %% [markdown]
# ## 5. K-Means

# %%
hasil_k = []
for k in range(2, 11):
    km_ = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    baris = {"k": k, "inertia": km_.inertia_, "silhouette": sil(X, km_.labels_),
             "davies_bouldin": davies_bouldin_score(X, km_.labels_)}
    if y is not None:
        baris["ARI_vs_label"] = adjusted_rand_score(y, km_.labels_)
    hasil_k.append(baris)
eval_k = pd.DataFrame(hasil_k)
print(eval_k.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3 if y is not None else 2, figsize=(15, 3.8))
ax[0].plot(eval_k.k, eval_k.inertia, "o-")
ax[0].set(title="Elbow (WCSS)", xlabel="k")
ax[1].plot(eval_k.k, eval_k.silhouette, "o-", color="#c55a11")
ax[1].set(title="Silhouette (sampel)", xlabel="k")
if y is not None:
    ax[2].plot(eval_k.k, eval_k.ARI_vs_label, "o-", color="#548235")
    ax[2].set(title="ARI terhadap label asli", xlabel="k")
fig.tight_layout()
simpan(fig, "05_kmeans_pemilihan_k.png")

# %%
# Tentukan k. Untuk "bean", bandingkan saran silhouette dengan jumlah varietas (7).
K = 7 if DATASET == "bean" else 4
t0 = time.time()
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X)
waktu = {"K-Means": time.time() - t0}
lab_km = km.labels_
print(f"K-Means k={K} selesai dalam {waktu['K-Means']:.1f} detik; ukuran klaster:", np.bincount(lab_km))

# %% [markdown]
# ## 6. Hierarchical clustering (Agglomerative)
# Hierarchical membutuhkan matriks jarak antar semua pasangan titik
# (kompleksitas memori O(n^2)). Dendrogram dibuat dari SAMPEL agar terbaca,
# sedangkan model akhir dijalankan pada seluruh data (atau sampel jika RAM terbatas).

# %%
HIER_SAMPLE = None        # isi 3000 bila RAM laptop < 8 GB

idx_d = rng.choice(len(X), 2000, replace=False)
Z = linkage(X[idx_d], method="ward")
fig, ax = plt.subplots(figsize=(13, 4.5))
dendrogram(Z, truncate_mode="lastp", p=40, ax=ax, color_threshold=Z[-(K - 1), 2])
ax.set(title=f"Dendrogram Ward (sampel 2.000 titik); garis warna = potongan {K} klaster",
       xlabel="Kelompok titik (ukuran dalam kurung)", ylabel="Jarak penggabungan")
simpan(fig, "06_dendrogram.png")

# Perbandingan metode linkage
idx_h = np.arange(len(X)) if HIER_SAMPLE is None else rng.choice(len(X), HIER_SAMPLE, replace=False)
banding_linkage = []
lab_hier = {}
for link in ["ward", "complete", "average"]:
    t0 = time.time()
    lab = AgglomerativeClustering(n_clusters=K, linkage=link).fit_predict(X[idx_h])
    dt = time.time() - t0
    lab_hier[link] = lab
    baris = {"linkage": link, "waktu_detik": round(dt, 1), "silhouette": sil(X[idx_h], lab),
             "ukuran_klaster": sorted(np.bincount(lab).tolist(), reverse=True)}
    if y is not None:
        baris["ARI_vs_label"] = adjusted_rand_score(y.iloc[idx_h], lab)
    banding_linkage.append(baris)
print(pd.DataFrame(banding_linkage).round(3).to_string(index=False))
waktu["Hierarchical (Ward)"] = banding_linkage[0]["waktu_detik"]

# %% [markdown]
# ## 7. DBSCAN
# Parameter eps dipilih dengan k-distance plot: jarak setiap titik ke tetangga
# ke-min_samples, diurutkan. "Lutut" kurva adalah kandidat eps.
# DBSCAN dijalankan pada 2 komponen PCA dan pada seluruh fitur untuk dibandingkan.

# %%
def kdist_plot(Xd, ms, judul, nama):
    d = np.sort(NearestNeighbors(n_neighbors=ms).fit(Xd).kneighbors(Xd)[0][:, -1])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(d)
    for q in (90, 95):
        ax.axhline(np.percentile(d, q), ls="--", color="grey")
        ax.text(5, np.percentile(d, q), f"persentil {q}", fontsize=8, va="bottom")
    ax.set(title=judul, xlabel="Titik (diurutkan)", ylabel=f"jarak ke tetangga ke-{ms}")
    simpan(fig, nama)


MS_PCA, MS_FULL = 40, 2 * n
kdist_plot(P2, MS_PCA, "k-distance (2 komponen PCA)", "07_kdist_pca2.png")
kdist_plot(X, MS_FULL, f"k-distance (seluruh {n} fitur)", "08_kdist_full.png")

grid = []
for ruang, Xd, ms, eps_list in [("PCA-2D", P2, MS_PCA, [0.15, 0.2, 0.25, 0.3, 0.4]),
                               (f"{n} fitur", X, MS_FULL, [0.8, 1.0, 1.5, 2.0, 2.5])]:
    for eps in eps_list:
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xd)
        baris = {"ruang": ruang, "eps": eps, "min_samples": ms,
                 "n_klaster": len(set(lab)) - (1 if -1 in lab else 0),
                 "noise_%": round((lab == -1).mean() * 100, 1)}
        if y is not None:
            baris["ARI_vs_label"] = round(adjusted_rand_score(y, lab), 3)
        grid.append(baris)
grid = pd.DataFrame(grid)
print(grid.to_string(index=False))

# %%
# Pilih kombinasi DBSCAN terbaik dari tabel di atas (ubah sesuai pengamatan Anda)
EPS_PILIH, RUANG_PILIH = (0.3, "pca") if DATASET == "bean" else (0.3, "pca")
Xd = P2 if RUANG_PILIH == "pca" else X
t0 = time.time()
lab_db = DBSCAN(eps=EPS_PILIH, min_samples=MS_PCA if RUANG_PILIH == "pca" else MS_FULL).fit_predict(Xd)
waktu["DBSCAN"] = time.time() - t0
print("DBSCAN: klaster =", len(set(lab_db)) - (1 if -1 in lab_db else 0),
      f"| noise = {(lab_db == -1).mean() * 100:.1f}%")

# %% [markdown]
# ## 8. Perbandingan ketiga algoritma

# %%
lab_ward = lab_hier["ward"] if HIER_SAMPLE is None else None
semua = {"K-Means": lab_km, "Hierarchical (Ward)": lab_ward, "DBSCAN": lab_db}
ringkas = []
for nama, lab in semua.items():
    if lab is None:
        continue
    m = lab != -1
    baris = {"algoritma": nama, "n_klaster": len(set(lab[m])),
             "noise_%": round((~m).mean() * 100, 1),
             "silhouette": round(sil(X, lab), 3),
             "davies_bouldin": round(davies_bouldin_score(X[m], lab[m]), 3) if len(set(lab[m])) > 1 else np.nan,
             "waktu_detik": round(waktu[nama], 1)}
    if y is not None:
        baris["ARI_vs_label"] = round(adjusted_rand_score(y, lab), 3)
    ringkas.append(baris)
ringkas = pd.DataFrame(ringkas)
print(ringkas.to_string(index=False))
ringkas.to_csv(os.path.join(OUT_DIR, "perbandingan_algoritma.csv"), index=False)

if lab_ward is not None:
    print("\nKesamaan K-Means vs Ward (ARI):", round(adjusted_rand_score(lab_km, lab_ward), 3))

panel = [("Label asli", y)] if y is not None else []
panel += [(k, v) for k, v in semua.items() if v is not None]
fig, axes = plt.subplots(1, len(panel), figsize=(5.2 * len(panel), 4.6))
for a, (judul, lab) in zip(axes, panel):
    lab = np.asarray(lab).astype(str)
    lab = np.where(lab == "-1", "noise", lab)
    urut = sorted(set(lab), key=lambda s: (s == "noise", s))
    pal = dict(zip([u for u in urut if u != "noise"], sns.color_palette("tab10", 10)))
    pal["noise"] = (0.8, 0.8, 0.8)
    sns.scatterplot(x=P2[:, 0], y=P2[:, 1], hue=lab, hue_order=urut, palette=pal,
                    s=4, linewidth=0, ax=a, legend=False)
    a.set(title=judul, xlabel="PC1", ylabel="PC2")
fig.tight_layout()
simpan(fig, "09_perbandingan_pca.png")

# %% [markdown]
# ## 9. Validasi eksternal (khusus dataset dengan label)

# %%
if y is not None:
    tab = pd.crosstab(y, lab_km, rownames=["Varietas asli"], colnames=["Klaster K-Means"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(tab, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Tabel silang varietas asli vs klaster K-Means")
    simpan(fig, "10_tabel_silang.png")
    purity = tab.max(axis=0).sum() / tab.values.sum()
    print(tab)
    print(f"\nPurity K-Means = {purity:.3f}  (proporsi titik yang masuk klaster bermayoritas varietasnya)")
    print("Varietas dominan tiap klaster:\n",
          (tab / tab.sum(axis=0)).idxmax().to_string(), "\n",
          (tab / tab.sum(axis=0)).max().round(2).to_string())

# %% [markdown]
# ## 10. Profil klaster (K-Means)

# %%
df["Klaster"] = lab_km
profil = df.groupby("Klaster")[FITUR].median()
profil.insert(0, "Jumlah", df["Klaster"].value_counts().sort_index())
profil.loc["Semua data"] = [len(df)] + df[FITUR].median().tolist()
print(profil.round(3).T.to_string())
profil.round(4).to_csv(os.path.join(OUT_DIR, "profil_klaster_kmeans.csv"))

zprof = pd.DataFrame(X, columns=FITUR).assign(Klaster=lab_km).groupby("Klaster").mean()
fig, ax = plt.subplots(figsize=(14, 0.55 * K + 2.5))
sns.heatmap(zprof, cmap="RdBu_r", center=0, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Profil klaster K-Means (rata-rata z-score)")
simpan(fig, "11_profil_klaster_heatmap.png")

df.to_csv(os.path.join(OUT_DIR, "hasil_klaster.csv"), index=False)
print(f"\nSelesai. Semua gambar dan tabel ada di folder {OUT_DIR}/")
