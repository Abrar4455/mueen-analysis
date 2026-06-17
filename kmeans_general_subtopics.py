"""
Mueen - Deep Dive into General Queries (Cluster 1 / 65%)
=========================================================
Extracts the general queries cluster and runs K-Means again
to discover hidden sub-topics within it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ─── PATHS ───────────────────────────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\abalbalushi\Downloads\Mueen_KMeans_Clustering.xlsx"
OUTPUT_FILE = r"C:\Users\abalbalushi\Downloads\Mueen_General_Queries_SubTopics.xlsx"
CHART_FILE  = r"C:\Users\abalbalushi\Downloads\Mueen_General_SubTopics_Chart.png"
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("DEEP DIVE — GENERAL QUERIES SUB-TOPIC ANALYSIS")
print("=" * 65)

# ─── 1. LOAD & FILTER GENERAL QUERIES (Cluster 1) ───────────────────────────
df = pd.read_excel(INPUT_FILE, sheet_name='Clustered Queries')
print(f"\n✅ Loaded {len(df)} total queries")

# Filter only Cluster 1 (the 65% general queries)
df_general = df[df['cluster_label'] == 'Cluster 1'].copy()
df_general = df_general[df_general['cleaned_query'].notna() & (df_general['cleaned_query'].str.strip() != '')]
print(f"   General queries (Cluster 1): {len(df_general)} queries ({len(df_general)/len(df)*100:.1f}%)")

# ─── 2. TF-IDF VECTORIZATION ─────────────────────────────────────────────────
print("\n📐 Step 1: Vectorizing general queries...")

vectorizer = TfidfVectorizer(
    max_features=300,
    min_df=2,
    max_df=0.85,
    analyzer='word',
    token_pattern=r'(?u)\b\w\w+\b'
)

X = vectorizer.fit_transform(df_general['cleaned_query'])
print(f"   Matrix: {X.shape[0]} queries × {X.shape[1]} features")

# ─── 3. FIND BEST K ──────────────────────────────────────────────────────────
print("\n📐 Step 2: Finding best number of sub-clusters...")

inertia_scores = []
silhouette_scores = []
K_range = range(2, 12)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia_scores.append(km.inertia_)
    sil = silhouette_score(X, km.labels_, sample_size=500, random_state=42)
    silhouette_scores.append(sil)
    print(f"   K={k:2d} | Silhouette: {sil:.4f}")

best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\n   ✅ Best sub-clusters: K = {best_k}")

# ─── 4. FINAL K-MEANS ────────────────────────────────────────────────────────
print(f"\n📐 Step 3: Running K-Means with K={best_k} on general queries...")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_general['sub_cluster'] = kmeans.fit_predict(X)

# ─── 5. EXTRACT TOP KEYWORDS PER SUB-CLUSTER ────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
sub_cluster_info = {}

print("\n" + "=" * 65)
print("DISCOVERED SUB-TOPICS WITHIN GENERAL QUERIES")
print("=" * 65)

for cid in range(best_k):
    centroid = kmeans.cluster_centers_[cid]
    top_indices = centroid.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    count = (df_general['sub_cluster'] == cid).sum()
    pct = count / len(df_general) * 100

    sub_cluster_info[cid] = {
        'label': f"Sub-Topic {cid + 1}",
        'keywords': top_words,
        'count': count,
        'pct': pct
    }

    print(f"\n🔵 Sub-Topic {cid + 1} ({count} queries, {pct:.1f}% of general)")
    print(f"   Keywords: {', '.join(top_words[:8])}")

df_general['sub_topic_label'] = df_general['sub_cluster'].apply(
    lambda x: sub_cluster_info[x]['label']
)

# ─── 6. CHARTS ───────────────────────────────────────────────────────────────
print("\n📊 Generating charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('General Queries — Hidden Sub-Topics (K-Means)', fontsize=14, fontweight='bold')

# Chart 1: Sub-topic sizes
counts = [sub_cluster_info[i]['count'] for i in range(best_k)]
labels = [f"Sub-Topic {i+1}\n{sub_cluster_info[i]['keywords'][0]}" for i in range(best_k)]
colors = plt.cm.Set2(np.linspace(0, 1, best_k))
axes[0].bar(labels, counts, color=colors)
axes[0].set_title('Queries per Sub-Topic')
axes[0].set_ylabel('Number of Queries')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(counts):
    axes[0].text(i, v + 3, str(v), ha='center', fontsize=9)

# Chart 2: 2D PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())
scatter_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cid in range(best_k):
    mask = df_general['sub_cluster'] == cid
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[scatter_colors[cid]],
                    label=f'Sub-Topic {cid+1}',
                    alpha=0.5, s=15)
axes[1].set_title('2D Visualization of Sub-Topics')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f"   ✅ Chart saved: {CHART_FILE}")

# ─── 7. SAVE EXCEL ───────────────────────────────────────────────────────────
print("\n💾 Saving results...")

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:

    # Sheet 1: All general queries with sub-topic labels
    df_general[['ministry', 'user_query', 'cleaned_query', 'sub_cluster', 'sub_topic_label', 'date']].to_excel(
        writer, sheet_name='General Queries SubTopics', index=False
    )

    # Sheet 2: Sub-topic summary
    summary = []
    for cid in range(best_k):
        info = sub_cluster_info[cid]
        summary.append({
            'Sub-Topic': info['label'],
            'Query Count': info['count'],
            'Percentage (%)': round(info['pct'], 1),
            'Top Keywords': ', '.join(info['keywords'][:8])
        })
    pd.DataFrame(summary).to_excel(writer, sheet_name='Sub-Topic Summary', index=False)

    # Sheet 3: Ministry breakdown by sub-topic
    ministry_sub = pd.crosstab(df_general['ministry'], df_general['sub_topic_label'])
    ministry_sub.to_excel(writer, sheet_name='Ministry by Sub-Topic')

print(f"\n✅ DONE! Saved: {OUTPUT_FILE}")
print(f"\n🎯 SUMMARY — Hidden topics inside the 65% general queries:")
for cid in range(best_k):
    info = sub_cluster_info[cid]
    print(f"   Sub-Topic {cid+1}: {info['count']} queries — {', '.join(info['keywords'][:4])}")
