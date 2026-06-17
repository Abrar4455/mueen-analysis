"""
============================================================
MUEEN CHATBOT — ANALYSIS PIPELINE WEB APP
============================================================
How to run:
  1. Install: pip install streamlit pandas scikit-learn matplotlib openpyxl
  2. Run:     streamlit run mueen_app.py
  3. Opens automatically in your browser!
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
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

# ─── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Mueen Chatbot Analysis",
    page_icon="🤖",
    layout="wide"
)

# ─── STYLING ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a3a5c;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #1a3a5c;
    }
    .step-header {
        background: #1a3a5c;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────
st.markdown('<div class="main-header">🤖 Mueen Chatbot — Analysis Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your dataset and get full topic modeling, clustering, and insights automatically</div>', unsafe_allow_html=True)
st.divider()

# ─── HELPER FUNCTIONS ────────────────────────────────────

TOPICS = {
    "📊 Statistical Data & Reports":  ["بيانات","احصاء","تقرير","نشرة","تعداد","مؤشر","statistics","data","report"],
    "🎓 Education":                    ["تعليم","طلبة","مدارس","جامعة","دراسة","قبول","education","students","school"],
    "👥 Employment & Labor":           ["مشتغلين","وظيفة","توظيف","عمل","موظف","تقاعد","employment","labor","job"],
    "🏛️ Ministry / Center Info":       ["المركز","الوطني","الاحصاء","وزارة","مديرية","ministry","center"],
    "⚖️ Legal & Regulations":          ["قانون","مرسوم","لائحة","نظام","قرار","law","decree","regulation"],
    "🌍 Population & Demographics":    ["سكان","محافظة","عمانيين","وافدين","تعداد","population","demographic"],
    "💼 Services & Procedures":        ["اجراءات","طلب","تقديم","ترخيص","رسوم","بطاقة","خدمة","service"],
    "💻 Digital Transformation":       ["رقمي","تحول","سحابية","نظام","digital","cloud","IT"],
    "🏥 Health":                       ["صحة","صحي","طبي","مستشفى","health","medical","hospital"],
    "🤖 About Mueen / AI":            ["معين","mueen","ذكاء","AI","chatbot","بوت"],
    "📝 Content / Translation":        ["ترجم","اكتب","صياغة","ملخص","translate","write","summarize"],
}

stop_words = {'من','في','على','الى','إلى','عن','ما','هو','هي','كم','هل','او','أو',
              'the','to','a','is','for','of','and','in','on','this','that','you','i',
              'مع','أن','التي','فيه','عليه'}

def clean_for_ml(text):
    text = re.sub(r'[^\u0600-\u06FF\s0-9a-zA-Z]', ' ', str(text))
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
    text = text.replace('ة','ه').replace('ى','ي')
    return re.sub(r'\s+', ' ', text).strip()

def classify_topic(text):
    if not isinstance(text, str): return "❓ Other"
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower) for t, kws in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "❓ Other"

def classify_intent(text):
    text = text.lower()
    if any(w in text for w in ["لخص","تلخيص","summary","summarize"]): return "Summarization"
    elif any(w in text for w in ["ترجم","translate"]): return "Translation"
    elif any(w in text for w in ["اكتب","write","خطاب","email"]): return "Email/Content Generation"
    else: return "Q&A / General"

def get_topic_name(keywords):
    kw = ' '.join(keywords)
    if any(w in kw for w in ['معين','برنامج','ai','mueen']): return '🤖 About Mueen & AI'
    elif any(w in kw for w in ['قانون','الحمايه','العمل','الاجتماعيه']): return '⚖️ Labor Law & Regulations'
    elif any(w in kw for w in ['ارض','الحصول','للحصول','الملف']): return '🏠 Land & Services'
    elif any(w in kw for w in ['اختصاصات','المشاريع','المعتمده']): return '🏛️ Ministry Roles'
    elif any(w in kw for w in ['للاحصاء','الوطني','المركز']): return '📊 National Statistics'
    elif any(w in kw for w in ['التعليم','طلاب']): return '🎓 Education'
    elif any(w in kw for w in ['how','can','what','is','the','mtcit']): return '🌍 English Queries'
    elif any(w in kw for w in ['تقع','ولايه','مسقط']): return '📍 Location & Governorate'
    elif any(w in kw for w in ['عدد','سكان','السلطنه']): return '👥 Population & Demographics'
    elif any(w in kw for w in ['تاريخ','اليوم','الوطني']): return '📅 Dates & Events'
    else: return '📋 General Government Info'

# ─── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Flag_of_Oman.svg/200px-Flag_of_Oman.svg.png", width=80)
    st.markdown("## ⚙️ Settings")
    max_k = st.slider("Max clusters to test (K)", min_value=5, max_value=20, value=12)
    show_raw = st.checkbox("Show raw data table", value=False)
    st.divider()
    st.markdown("**📌 How to use:**")
    st.markdown("1. Upload your Excel file\n2. Click **Run Analysis**\n3. View results\n4. Download report")
    st.divider()
    st.markdown("*OTech Oman — AI Department*")

# ─── FILE UPLOAD ─────────────────────────────────────────
st.markdown("### 📂 Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload your Mueen chatbot Excel file (.xlsx or .xlsm)",
    type=["xlsx", "xlsm"],
    help="Must contain 'user_query' and 'ministry' columns"
)

if uploaded_file:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")

    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):

        # ── LOAD ──────────────────────────────────────────
        with st.spinner("Loading data..."):
            df = pd.read_excel(uploaded_file)
            query_col  = [c for c in df.columns if 'query' in c.lower() or 'سؤال' in c][0]
            domain_col = [c for c in df.columns if 'ministry' in c.lower() or 'domain' in c.lower() or 'جهة' in c][0]
            df[query_col]  = df[query_col].astype(str).fillna('')
            df[domain_col] = df[domain_col].astype(str).fillna('Unknown')

        # ── CLEAN ─────────────────────────────────────────
        with st.spinner("Cleaning data..."):
            greetings = ["hi","hello","مرحبا","السلام عليكم","اهلا","هلا","good morning","good evening","حيالله"]
            df['word_count'] = df[query_col].apply(lambda t: len(t.split()))
            df[query_col] = df[query_col].apply(lambda t: "" if t.lower().strip() in greetings else t)
            df = df[df[query_col].str.strip() != ""].reset_index(drop=True)
            df['cleaned_query'] = df[query_col].apply(clean_for_ml)
            df['topic']  = df[query_col].apply(classify_topic)
            df['intent'] = df[query_col].apply(classify_intent)
            total = len(df)

        # ── METRICS ───────────────────────────────────────
        st.divider()
        st.markdown("## 📊 Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", f"{total:,}")
        c2.metric("Ministries", df[domain_col].nunique())
        c3.metric("Avg Words/Query", f"{df['word_count'].mean():.1f}")
        c4.metric("Total Words", f"{df['word_count'].sum():,}")

        # ── TOPIC RESULTS ─────────────────────────────────
        st.divider()
        st.markdown("## 🏷️ Topic Classification")

        col1, col2 = st.columns(2)

        with col1:
            topic_counts = df['topic'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            topic_counts['%'] = (topic_counts['Count'] / total * 100).round(1)
            st.dataframe(topic_counts, use_container_width=True, hide_index=True)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            top5 = topic_counts.head(8)
            colors = plt.cm.Set3(np.linspace(0, 1, len(top5)))
            ax.barh(top5['Topic'], top5['Count'], color=colors)
            ax.set_xlabel('Number of Queries')
            ax.set_title('Top Topics')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── INTENT ────────────────────────────────────────
        st.divider()
        st.markdown("## 🎯 Intent Classification")
        col3, col4 = st.columns(2)

        intent_counts = df['intent'].value_counts()
        with col3:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(intent_counts.values, labels=intent_counts.index,
                    autopct='%1.1f%%', colors=plt.cm.Pastel1(np.linspace(0,1,len(intent_counts))))
            ax2.set_title('Intent Distribution')
            st.pyplot(fig2)
            plt.close()

        with col4:
            intent_ministry = pd.crosstab(df[domain_col], df['intent'])
            st.markdown("**Intent by Ministry (Top 10)**")
            st.dataframe(intent_ministry.head(10), use_container_width=True)

        # ── K-MEANS ───────────────────────────────────────
        st.divider()
        st.markdown("## 🔵 K-Means Clustering")

        with st.spinner(f"Running K-Means (testing K=2 to {max_k})..."):
            df_ml = df[df['cleaned_query'].str.len() > 2].copy()
            vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.85,
                                          analyzer='word', token_pattern=r'(?u)\b\w\w+\b')
            X = vectorizer.fit_transform(df_ml['cleaned_query'])

            inertia_list, sil_list = [], []
            K_range = range(2, max_k + 1)
            progress = st.progress(0)
            for i, k in enumerate(K_range):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inertia_list.append(km.inertia_)
                sil = silhouette_score(X, km.labels_, sample_size=min(500, X.shape[0]), random_state=42)
                sil_list.append(sil)
                progress.progress((i+1)/len(K_range))

            best_k = K_range[sil_list.index(max(sil_list))]
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            df_ml['cluster'] = kmeans.fit_predict(X)
            feature_names = vectorizer.get_feature_names_out()

            cluster_info = {}
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

        st.success(f"✅ Best K = **{best_k}** clusters found!")

        col5, col6 = st.columns(2)
        with col5:
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.plot(list(K_range), inertia_list, 'bo-')
            ax3.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
            ax3.set_title('Elbow Method'); ax3.set_xlabel('K'); ax3.set_ylabel('Inertia')
            ax3.legend(); ax3.grid(True, alpha=0.3)
            st.pyplot(fig3); plt.close()

        with col6:
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            ax4.plot(list(K_range), sil_list, 'gs-')
            ax4.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
            ax4.set_title('Silhouette Score'); ax4.set_xlabel('K'); ax4.set_ylabel('Score')
            ax4.legend(); ax4.grid(True, alpha=0.3)
            st.pyplot(fig4); plt.close()

        # Cluster summary table
        cluster_df = pd.DataFrame([{
            'Topic': cluster_info[cid]['topic_name'],
            'Queries': cluster_info[cid]['count'],
            '%': round(cluster_info[cid]['pct'], 1),
            'Top Keywords': ', '.join(cluster_info[cid]['keywords'][:5])
        } for cid in range(best_k)]).sort_values('%', ascending=False)

        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

        # 2D PCA chart
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X.toarray())
        sc_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
        for cid in range(best_k):
            mask = df_ml['cluster'] == cid
            ax5.scatter(X_2d[mask,0], X_2d[mask,1], c=[sc_colors[cid]],
                        label=cluster_info[cid]['topic_name'][:25], alpha=0.4, s=10)
        ax5.set_title('2D Cluster Visualization (PCA)')
        ax5.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        st.pyplot(fig5); plt.close()

        # ── MINISTRY BREAKDOWN ────────────────────────────
        st.divider()
        st.markdown("## 🏢 Ministry Breakdown")
        ministry_summary = []
        for ministry in df[domain_col].unique():
            min_df = df[df[domain_col] == ministry]
            all_words = re.findall(r'\b\w+\b', ' '.join(min_df[query_col]).lower())
            meaningful = [w for w in all_words if w not in stop_words and not w.isdigit()]
            top_words = ', '.join([f"{w}({c})" for w,c in Counter(meaningful).most_common(3)])
            ministry_summary.append({
                'Ministry': ministry,
                'Queries': len(min_df),
                'Volume %': round(len(min_df)/total*100, 1),
                'Avg Words': round(min_df['word_count'].mean(), 1),
                'Top Keywords': top_words
            })
        ministry_df = pd.DataFrame(ministry_summary).sort_values('Queries', ascending=False)
        st.dataframe(ministry_df, use_container_width=True, hide_index=True)

        # ── RAW DATA ──────────────────────────────────────
        if show_raw:
            st.divider()
            st.markdown("## 📋 Raw Data")
            st.dataframe(df[[domain_col, query_col, 'word_count', 'topic', 'intent']].head(100),
                         use_container_width=True, hide_index=True)

        # ── DOWNLOAD REPORT ───────────────────────────────
        st.divider()
        st.markdown("## 💾 Download Report")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df[[domain_col, query_col, 'word_count', 'topic', 'intent']].to_excel(
                writer, sheet_name='All Queries', index=False)
            pd.DataFrame(ministry_summary).to_excel(writer, sheet_name='Dashboard', index=False)
            topic_counts.to_excel(writer, sheet_name='Topic Summary', index=False)
            cluster_df.to_excel(writer, sheet_name='Cluster Summary', index=False)
            intent_ministry.to_excel(writer, sheet_name='Intent by Ministry')
            for ministry in df[domain_col].unique():
                min_df = df[df[domain_col] == ministry]
                min_df[[query_col, 'word_count', 'topic', 'intent']].to_excel(
                    writer, sheet_name=str(ministry)[:30], index=False)

        output.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Full Excel Report",
            data=output,
            file_name=f"Mueen_Analysis_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.success("🎉 Analysis complete! Download your report above.")

else:
    st.info("👆 Please upload your Mueen chatbot Excel file to get started.")
    st.markdown("""
    ### What this app does:
    - ✅ **Cleans** your data automatically
    - ✅ **Classifies topics** using keyword matching
    - ✅ **Runs K-Means clustering** to discover hidden topics
    - ✅ **Shows charts** — elbow curve, topic distribution, PCA visualization
    - ✅ **Ministry breakdown** with top keywords per ministry
    - ✅ **Downloadable Excel report** with all results
    """)
