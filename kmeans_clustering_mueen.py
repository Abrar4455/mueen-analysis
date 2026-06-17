"""
Mueen Chatbot - Advanced Topic Modeling with K-Means Clustering
================================================================
Instead of manually defining topics with keywords,
K-Means automatically discovers hidden topics from the data.

Libraries used:
- sklearn (TF-IDF vectorizer + KMeans)
- matplotlib / seaborn (charts)
- pandas (data handling)
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ─── PATHS — UPDATE IF NEEDED ───────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\abalbalushi\Downloads\mueen_topic_modeling_results.xlsx"
OUTPUT_FILE = r"C:\Users\abalbalushi\Downloads\Mueen_KMeans_Clustering.xlsx"
CHART_FILE  = r"C:\Users\abalbalushi\Downloads\Mueen_Clusters_Chart.png"
# ────────────────────────────────────────────────────────────────────────────

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("=" * 65)
print("MUEEN CHATBOT — K-MEANS CLUSTERING (Advanced Topic Modeling)")
print("=" * 65)

df = pd.read_excel(INPUT_FILE, sheet_name='Labeled Queries')
print(f"\n✅ Loaded {len(df)} queries")

# Use cleaned_query column for clustering
df = df[df['cleaned_query'].notna() & (df['cleaned_query'].str.strip() != '')]
print(f"   After removing empty rows: {len(df)} queries")

# ─── 2. TEXT VECTORIZATION (TF-IDF) ─────────────────────────────────────────
# TF-IDF converts text into numbers so K-Means can process it.
# It gives higher scores to words that are IMPORTANT in a query
# but not too common across all queries.

print("\n📐 Step 1: Vectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=500,       # use top 500 most important words
    min_df=2,               # ignore words appearing in fewer than 2 docs
    max_df=0.85,            # ignore words appearing in more than 85% of docs
    analyzer='word',
    token_pattern=r'(?u)\b\w\w+\b'  # words with 2+ characters
)

X = vectorizer.fit_transform(df['cleaned_query'])
print(f"   Text matrix shape: {X.shape[0]} queries × {X.shape[1]} features")

# ─── 3. FIND BEST NUMBER OF CLUSTERS (Elbow Method) ─────────────────────────
# We try K = 2 to 12 and find the "elbow point" where adding
# more clusters stops improving the model significantly.

print("\n📐 Step 2: Finding best number of clusters (Elbow Method)...")

inertia_scores = []
silhouette_scores = []
K_range = range(2, 13)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia_scores.append(km.inertia_)
    sil = silhouette_score(X, km.labels_, sample_size=500, random_state=42)
    silhouette_scores.append(sil)
    print(f"   K={k:2d} | Inertia: {km.inertia_:,.0f} | Silhouette: {sil:.4f}")

# Best K = highest silhouette score
best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\n   ✅ Best number of clusters: K = {best_k}")

# ─── 4. FINAL K-MEANS MODEL ──────────────────────────────────────────────────
print(f"\n📐 Step 3: Running K-Means with K={best_k}...")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# ─── 5. LABEL EACH CLUSTER WITH TOP KEYWORDS ────────────────────────────────
# For each cluster, find the top TF-IDF words — these tell us the topic

print("\n📐 Step 4: Extracting top keywords per cluster...")

feature_names = vectorizer.get_feature_names_out()
cluster_labels = {}
cluster_keywords = {}

print("\n" + "=" * 65)
print("DISCOVERED CLUSTERS (Auto-detected topics)")
print("=" * 65)

for cluster_id in range(best_k):
    # Get the centroid (average position) of this cluster
    centroid = kmeans.cluster_centers_[cluster_id]
    # Top 10 words with highest TF-IDF in this cluster
    top_indices = centroid.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]

    cluster_keywords[cluster_id] = top_words
    count = (df['cluster'] == cluster_id).sum()
    pct = count / len(df) * 100

    # Auto-label based on top words
    label = f"Cluster {cluster_id + 1}"
    cluster_labels[cluster_id] = label

    print(f"\n🔵 {label} ({count} queries, {pct:.1f}%)")
    print(f"   Top keywords: {', '.join(top_words[:8])}")

# ─── 6. ADD CLUSTER LABEL TO DATAFRAME ──────────────────────────────────────
df['cluster_label'] = df['cluster'].map(cluster_labels)

# ─── 7. COMPARE WITH ORIGINAL KEYWORD TOPICS ────────────────────────────────
print("\n" + "=" * 65)
print("CLUSTER vs ORIGINAL TOPIC COMPARISON")
print("=" * 65)
comparison = pd.crosstab(df['cluster_label'], df['topic_english'])
print(comparison.to_string())

# ─── 8. CHARTS ───────────────────────────────────────────────────────────────
print("\n📊 Step 5: Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Mueen Chatbot — K-Means Clustering Results', fontsize=16, fontweight='bold')

# Chart 1: Elbow curve
axes[0, 0].plot(list(K_range), inertia_scores, 'bo-', linewidth=2, markersize=6)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0, 0].set_title('Elbow Method — Finding Best K')
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Chart 2: Silhouette scores
axes[0, 1].plot(list(K_range), silhouette_scores, 'gs-', linewidth=2, markersize=6)
axes[0, 1].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0, 1].set_title('Silhouette Score — Cluster Quality')
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Chart 3: Cluster sizes (bar chart)
cluster_counts = df['cluster_label'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color=colors)
axes[1, 0].set_title(f'Queries per Cluster (K={best_k})')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Number of Queries')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(cluster_counts.values):
    axes[1, 0].text(i, v + 5, str(v), ha='center', fontsize=9)

# Chart 4: 2D PCA visualization of clusters
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())
scatter_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cluster_id in range(best_k):
    mask = df['cluster'] == cluster_id
    axes[1, 1].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=[scatter_colors[cluster_id]],
        label=f'C{cluster_id + 1}',
        alpha=0.5, s=15
    )
axes[1, 1].set_title('2D Cluster Visualization (PCA)')
axes[1, 1].set_xlabel('PCA Component 1')
axes[1, 1].set_ylabel('PCA Component 2')
axes[1, 1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
print(f"   ✅ Chart saved: {CHART_FILE}")

# ─── 9. SAVE EXCEL RESULTS ───────────────────────────────────────────────────
print("\n💾 Step 6: Saving Excel results...")

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:

    # Sheet 1: Full labeled dataset with cluster assignments
    df[['ministry', 'user_query', 'cleaned_query',
        'topic_english', 'cluster', 'cluster_label', 'date']].to_excel(
        writer, sheet_name='Clustered Queries', index=False
    )

    # Sheet 2: Cluster summary with top keywords
    cluster_summary = []
    for cluster_id in range(best_k):
        count = (df['cluster'] == cluster_id).sum()
        cluster_summary.append({
            'Cluster': cluster_labels[cluster_id],
            'Query Count': count,
            'Percentage (%)': round(count / len(df) * 100, 1),
            'Top Keywords': ', '.join(cluster_keywords[cluster_id][:8])
        })
    pd.DataFrame(cluster_summary).to_excel(
        writer, sheet_name='Cluster Summary', index=False
    )

    # Sheet 3: Cluster vs Original Topic comparison
    comparison.to_excel(writer, sheet_name='Cluster vs Topic')

    # Sheet 4: Ministry breakdown by cluster
    ministry_cluster = pd.crosstab(df['ministry'], df['cluster_label'])
    ministry_cluster.to_excel(writer, sheet_name='Ministry by Cluster')

    # Sheet 5: Elbow data
    elbow_df = pd.DataFrame({
        'K': list(K_range),
        'Inertia': inertia_scores,
        'Silhouette Score': silhouette_scores
    })
    elbow_df.to_excel(writer, sheet_name='Elbow Data', index=False)

print(f"\n✅ Excel saved: {OUTPUT_FILE}")
print(f"\n🎯 SUMMARY:")
print(f"   Best number of clusters (topics): K = {best_k}")
print(f"   Total queries clustered: {len(df)}")
for cluster_id in range(best_k):
    count = (df['cluster'] == cluster_id).sum()
    print(f"   {cluster_labels[cluster_id]}: {count} queries — top words: {', '.join(cluster_keywords[cluster_id][:5])}")
