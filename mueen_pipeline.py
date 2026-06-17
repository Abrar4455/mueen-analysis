"""
============================================================
MUEEN CHATBOT — FULL ANALYSIS PIPELINE
============================================================
How to use:
1. Set INPUT_FILE to your new Excel/xlsm file path
2. Run the script
3. Find all results in the OUTPUT_FOLDER

What it does automatically:
  Step 1 → Load & clean data
  Step 2 → Topic classification (keyword-based)
  Step 3 → K-Means clustering (auto topic discovery)
  Step 4 → Sub-topic analysis (deep dive into general queries)
  Step 5 → Save Excel reports + charts
============================================================
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════╗
# ║              ⚙️  SETTINGS — EDIT THESE              ║
# ╚══════════════════════════════════════════════════════╝

INPUT_FILE    = r"C:\Users\abalbalushi\Downloads\RAG Chatbot Data - Mueen.xlsm"
OUTPUT_FOLDER = r"C:\Users\abalbalushi\Downloads\Mueen_Pipeline_Output"

# ════════════════════════════════════════════════════════

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def log(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")
def info(msg): print(f"   ➤ {msg}")

# ─────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────
log("STEP 1: Loading Data")

df = pd.read_excel(INPUT_FILE)
info(f"Loaded {len(df)} rows")
info(f"Columns: {list(df.columns)}")

# Auto-detect columns
query_col  = [c for c in df.columns if 'query' in c.lower() or 'text' in c.lower() or 'سؤال' in c][0]
domain_col = [c for c in df.columns if 'ministry' in c.lower() or 'domain' in c.lower() or 'جهة' in c][0]

info(f"Query column: '{query_col}'")
info(f"Ministry column: '{domain_col}'")

df[query_col]  = df[query_col].astype(str).fillna('')
df[domain_col] = df[domain_col].astype(str).fillna('Unknown')

# ─────────────────────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────────────────────
log("STEP 2: Cleaning Data")

# Word count before cleaning
df['word_count'] = df[query_col].apply(lambda t: len(t.split()))
total_records = len(df)
info(f"Total records: {total_records}")
info(f"Total words: {df['word_count'].sum()}")
info(f"Average words per query: {df['word_count'].mean():.2f}")

# Remove greetings
greetings = [
    "hi", "hello", "مرحبا", "السلام عليكم", "اهلا", "هلا",
    "good morning", "good evening", "حيالله", "صباح الخير", "مساء الخير"
]

def remove_greetings(text):
    return "" if text.lower().strip() in greetings else text

df[query_col] = df[query_col].apply(remove_greetings)
df = df[df[query_col].str.strip() != ""].reset_index(drop=True)
info(f"After cleaning: {len(df)} rows remaining")

# Cleaned text for ML
def clean_for_ml(text):
    text = re.sub(r'[^\u0600-\u06FF\s0-9a-zA-Z]', ' ', str(text))
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
    text = text.replace('ة','ه').replace('ى','ي')
    return re.sub(r'\s+', ' ', text).strip()

df['cleaned_query'] = df[query_col].apply(clean_for_ml)

# ─────────────────────────────────────────────────────────
# STEP 3: KEYWORD-BASED TOPIC CLASSIFICATION
# ─────────────────────────────────────────────────────────
log("STEP 3: Keyword-Based Topic Classification")

TOPICS = {
    "📊 Statistical Data & Reports":      ["بيانات", "احصاء", "احصائية", "تقرير", "نشرة", "تعداد", "مؤشر", "statistics", "data", "report"],
    "🎓 Education":                        ["تعليم", "طلبة", "مدارس", "جامعة", "دراسة", "قبول", "education", "students", "school"],
    "👥 Employment & Labor":               ["مشتغلين", "وظيفة", "توظيف", "عمل", "موظف", "تقاعد", "employment", "labor", "job"],
    "🏛️ Ministry / Center Info":           ["المركز", "الوطني", "الاحصاء", "وزارة", "مديرية", "تعريف", "ministry", "center"],
    "⚖️ Legal & Regulations":              ["قانون", "مرسوم", "لائحة", "نظام", "قرار", "تشريع", "law", "decree", "regulation"],
    "🌍 Population & Demographics":        ["سكان", "محافظة", "عمانيين", "وافدين", "تعداد", "population", "demographic"],
    "💼 Services & Procedures":            ["اجراءات", "طلب", "تقديم", "ترخيص", "رسوم", "بطاقة", "خدمة", "procedure", "service"],
    "💻 Digital Transformation":           ["رقمي", "تحول", "سحابية", "نظام", "digital", "cloud", "IT"],
    "🏥 Health":                           ["صحة", "صحي", "طبي", "مستشفى", "health", "medical", "hospital"],
    "📦 Trade & Economy":                  ["واردات", "صادرات", "تجاري", "اقتصاد", "trade", "imports", "exports"],
    "🚗 Traffic & Transportation":         ["مرور", "مخالفة", "حوادث", "سيارة", "traffic", "violation", "accident"],
    "🤖 About Mueen / AI":                ["معين", "mueen", "ذكاء", "AI", "chatbot", "بوت"],
    "📝 Content / Translation":            ["ترجم", "اكتب", "صياغة", "ملخص", "translate", "write", "summarize"],
}

def classify_topic(text):
    if not isinstance(text, str): return "❓ Other"
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower) for t, kws in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "❓ Other"

# Intent classification
def classify_intent(text):
    text = text.lower()
    if any(w in text for w in ["لخص","تلخيص","summary","summarize"]): return "Summarization"
    elif any(w in text for w in ["ترجم","translate"]): return "Translation"
    elif any(w in text for w in ["اكتب","write","خطاب","email"]): return "Email/Content Generation"
    else: return "Q&A / General"

df['topic']  = df[query_col].apply(classify_topic)
df['intent'] = df[query_col].apply(classify_intent)

topic_counts = df['topic'].value_counts()
info("Top topics found:")
for topic, count in topic_counts.head(5).items():
    info(f"  {topic}: {count} queries ({count/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────
# STEP 4: K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────
log("STEP 4: K-Means Clustering (Auto Topic Discovery)")

df_ml = df[df['cleaned_query'].str.len() > 2].copy()

vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.85,
                              analyzer='word', token_pattern=r'(?u)\b\w\w+\b')
X = vectorizer.fit_transform(df_ml['cleaned_query'])
info(f"TF-IDF matrix: {X.shape[0]} queries × {X.shape[1]} features")

# Find best K
info("Finding best K (testing K=2 to 12)...")
inertia_scores, silhouette_scores_list = [], []
K_range = range(2, 13)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia_scores.append(km.inertia_)
    sil = silhouette_score(X, km.labels_, sample_size=min(500, X.shape[0]), random_state=42)
    silhouette_scores_list.append(sil)
    info(f"K={k:2d} | Silhouette: {sil:.4f}")

best_k = K_range[silhouette_scores_list.index(max(silhouette_scores_list))]
info(f"✅ Best K = {best_k}")

# Final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_ml['cluster'] = kmeans.fit_predict(X)

# Label clusters
feature_names = vectorizer.get_feature_names_out()
cluster_info = {}

def get_topic_name(keywords):
    kw = ' '.join(keywords)
    if any(w in kw for w in ['معين','برنامج','ai','منصه','mueen']): return '🤖 About Mueen & AI'
    elif any(w in kw for w in ['قانون','الحمايه','العمل','الاجتماعيه']): return '⚖️ Labor Law & Regulations'
    elif any(w in kw for w in ['ارض','الحصول','للحصول','الملف']): return '🏠 Land & Services Procedures'
    elif any(w in kw for w in ['اختصاصات','المشاريع','المعتمده']): return '🏛️ Ministry Roles & Info'
    elif any(w in kw for w in ['للاحصاء','الوطني','المركز']): return '📊 National Statistics'
    elif any(w in kw for w in ['التعليم','الجهه','طلاب']): return '🎓 Education'
    elif any(w in kw for w in ['how','can','mtcit','does','what','is','the']): return '🌍 English Language Queries'
    elif any(w in kw for w in ['تقع','ولايه','مسقط']): return '📍 Location & Governorate'
    elif any(w in kw for w in ['موقع','الوزاره','التواصل']): return '🌐 Ministry Website & Contact'
    elif any(w in kw for w in ['عدد','سكان','السلطنه']): return '👥 Population & Demographics'
    elif any(w in kw for w in ['تاريخ','اليوم','الوطني']): return '📅 Dates & Events'
    else: return '📋 General Government Info'

for cid in range(best_k):
    centroid = kmeans.cluster_centers_[cid]
    top_words = [feature_names[i] for i in centroid.argsort()[-10:][::-1]]
    count = (df_ml['cluster'] == cid).sum()
    cluster_info[cid] = {
        'topic_name': get_topic_name(top_words),
        'keywords': top_words,
        'count': count,
        'pct': count / len(df_ml) * 100
    }

df_ml['cluster_label'] = df_ml['cluster'].apply(lambda x: cluster_info[x]['topic_name'])

info("Clusters discovered:")
for cid, info_dict in cluster_info.items():
    info(f"  {info_dict['topic_name']}: {info_dict['count']} queries ({info_dict['pct']:.1f}%)")

# ─────────────────────────────────────────────────────────
# STEP 5: SUB-TOPIC ANALYSIS (General Queries Deep Dive)
# ─────────────────────────────────────────────────────────
log("STEP 5: Sub-Topic Analysis on General Queries")

general_label = cluster_info[max(cluster_info, key=lambda x: cluster_info[x]['count'])]['topic_name']
df_general = df_ml[df_ml['cluster_label'] == general_label].copy()
info(f"General queries cluster: '{general_label}' — {len(df_general)} queries")

X_gen = vectorizer.transform(df_general['cleaned_query'])

sil_gen = []
K_gen = range(2, 12)
for k in K_gen:
    km_g = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_g.fit(X_gen)
    sil_gen.append(silhouette_score(X_gen, km_g.labels_, sample_size=min(500, X_gen.shape[0]), random_state=42))

best_k_gen = K_gen[sil_gen.index(max(sil_gen))]
info(f"Best sub-clusters K = {best_k_gen}")

km_final = KMeans(n_clusters=best_k_gen, random_state=42, n_init=10)
df_general['sub_cluster'] = km_final.fit_predict(X_gen)

sub_info = {}
for cid in range(best_k_gen):
    centroid = km_final.cluster_centers_[cid]
    top_words = [feature_names[i] for i in centroid.argsort()[-10:][::-1]]
    count = (df_general['sub_cluster'] == cid).sum()
    sub_info[cid] = {
        'topic_name': get_topic_name(top_words),
        'keywords': top_words,
        'count': count,
        'pct': count / len(df_general) * 100
    }

df_general['sub_topic_label'] = df_general['sub_cluster'].apply(lambda x: sub_info[x]['topic_name'])

# ─────────────────────────────────────────────────────────
# STEP 6: GENERATE CHARTS
# ─────────────────────────────────────────────────────────
log("STEP 6: Generating Charts")

# Chart 1: K-Means main chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Mueen Chatbot — K-Means Clustering Results', fontsize=16, fontweight='bold')

axes[0,0].plot(list(K_range), inertia_scores, 'bo-', linewidth=2)
axes[0,0].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0,0].set_title('Elbow Method')
axes[0,0].set_xlabel('K'); axes[0,0].set_ylabel('Inertia')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(list(K_range), silhouette_scores_list, 'gs-', linewidth=2)
axes[0,1].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0,1].set_title('Silhouette Score')
axes[0,1].set_xlabel('K'); axes[0,1].set_ylabel('Score')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

cluster_counts = df_ml['cluster_label'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
axes[1,0].barh(cluster_counts.index, cluster_counts.values, color=colors)
axes[1,0].set_title('Queries per Topic')
axes[1,0].set_xlabel('Count')

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())
sc_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cid in range(best_k):
    mask = df_ml['cluster'] == cid
    axes[1,1].scatter(X_2d[mask,0], X_2d[mask,1], c=[sc_colors[cid]],
                      label=cluster_info[cid]['topic_name'][:20], alpha=0.4, s=10)
axes[1,1].set_title('2D PCA Visualization')
axes[1,1].legend(fontsize=6, loc='upper right')

plt.tight_layout()
chart1_path = os.path.join(OUTPUT_FOLDER, f'KMeans_Chart_{timestamp}.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
info(f"Chart 1 saved: {chart1_path}")

# Chart 2: Sub-topics pie chart
fig2, ax2 = plt.subplots(figsize=(10, 7))
sub_counts = df_general['sub_topic_label'].value_counts()
colors2 = plt.cm.Set2(np.linspace(0, 1, len(sub_counts)))
ax2.pie(sub_counts.values, labels=[f"{l}\n({v})" for l, v in zip(sub_counts.index, sub_counts.values)],
        colors=colors2, autopct='%1.1f%%', startangle=90)
ax2.set_title('Sub-Topics within General Queries', fontsize=13, fontweight='bold')
plt.tight_layout()
chart2_path = os.path.join(OUTPUT_FOLDER, f'SubTopics_Chart_{timestamp}.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
info(f"Chart 2 saved: {chart2_path}")

# ─────────────────────────────────────────────────────────
# STEP 7: SAVE EXCEL REPORTS
# ─────────────────────────────────────────────────────────
log("STEP 7: Saving Excel Reports")

# --- Report 1: Statistical Report ---
stop_words = {'من','في','على','الى','إلى','عن','ما','هو','هي','كم','هل','او','أو',
              'the','to','a','is','for','of','and','in','on','this','that','you','i',
              'مع','أن','التي','فيه','عليه'}

intent_by_ministry = pd.crosstab(df[domain_col], df['intent'])
summary_data = []

stat_path = os.path.join(OUTPUT_FOLDER, f'Mueen_Statistical_Report_{timestamp}.xlsx')
with pd.ExcelWriter(stat_path, engine='openpyxl') as writer:
    for ministry in df[domain_col].unique():
        min_df = df[df[domain_col] == ministry]
        all_words = re.findall(r'\b\w+\b', ' '.join(min_df[query_col]).lower())
        meaningful = [w for w in all_words if w not in stop_words and not w.isdigit()]
        top_words = ', '.join([f"{w}({c})" for w, c in Counter(meaningful).most_common(5)])
        summary_data.append({
            'Ministry': ministry,
            'Total Records': len(min_df),
            'Volume %': round(len(min_df)/total_records*100, 2),
            'Total Words': min_df['word_count'].sum(),
            'Avg Words': round(min_df['word_count'].mean(), 2),
            'Top Keywords': top_words,
            'Main Intent': intent_by_ministry.loc[ministry].idxmax() if ministry in intent_by_ministry.index else 'N/A'
        })
        min_df[[query_col, 'word_count', 'topic', 'intent']].to_excel(
            writer, sheet_name=str(ministry)[:30], index=False)

    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Dashboard', index=False)
    df[[domain_col, query_col, 'word_count', 'topic', 'intent']].to_excel(
        writer, sheet_name='All Queries', index=False)
    intent_by_ministry.to_excel(writer, sheet_name='Intent by Ministry')
    pd.DataFrame(topic_counts).reset_index().to_excel(writer, sheet_name='Topic Summary', index=False)

info(f"Statistical report saved: {stat_path}")

# --- Report 2: Clustering Report ---
cluster_path = os.path.join(OUTPUT_FOLDER, f'Mueen_Clustering_Report_{timestamp}.xlsx')
with pd.ExcelWriter(cluster_path, engine='openpyxl') as writer:
    df_ml[['ministry' if 'ministry' in df_ml.columns else domain_col,
           query_col, 'cleaned_query', 'topic', 'cluster_label']].to_excel(
        writer, sheet_name='Clustered Queries', index=False)

    cluster_summary = [{
        'Topic Name': cluster_info[cid]['topic_name'],
        'Query Count': cluster_info[cid]['count'],
        'Percentage (%)': round(cluster_info[cid]['pct'], 1),
        'Top Keywords': ', '.join(cluster_info[cid]['keywords'][:6])
    } for cid in range(best_k)]
    pd.DataFrame(cluster_summary).to_excel(writer, sheet_name='Cluster Summary', index=False)

    df_general[['sub_topic_label', 'sub_cluster']].value_counts().reset_index().to_excel(
        writer, sheet_name='Sub-Topic Summary', index=False)

    df_general[[query_col, 'sub_topic_label']].to_excel(
        writer, sheet_name='General Queries Detail', index=False)

    pd.DataFrame({
        'K': list(K_range),
        'Inertia': inertia_scores,
        'Silhouette Score': silhouette_scores_list
    }).to_excel(writer, sheet_name='Elbow Data', index=False)

info(f"Clustering report saved: {cluster_path}")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
log("✅ PIPELINE COMPLETE!")
print(f"""
📁 Output folder: {OUTPUT_FOLDER}
📊 Files created:
   • Mueen_Statistical_Report_{timestamp}.xlsx
   • Mueen_Clustering_Report_{timestamp}.xlsx
   • KMeans_Chart_{timestamp}.png
   • SubTopics_Chart_{timestamp}.png

🎯 Key Results:
   • Total queries analyzed : {len(df)}
   • Best K (clusters)      : {best_k}
   • Top topic              : {topic_counts.index[0]} ({topic_counts.iloc[0]} queries)
   • General queries        : {len(df_general)} ({len(df_general)/len(df_ml)*100:.1f}%)
   • Sub-topics found       : {best_k_gen}
""")
