import pandas as pd
import re
from collections import Counter
from deep_translator import GoogleTranslator
import time

# =========================
# ✅ TRANSLATION HELPER
# =========================

def translate_to_english(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
    if not has_arabic:
        return text  # Already English
    try:
        return GoogleTranslator(source='ar', target='en').translate(text[:4500])
    except:
        return "[Translation error]"

# =========================
# 1. Load dataset
# =========================

file_path = r"C:\Users\abalbalushi\Downloads\RAG Chatbot Data - Mueen.xlsm"
df = pd.read_excel(file_path)

# Print columns
print("Your Excel columns are named:", list(df.columns))

# Detect columns
query_col = [col for col in df.columns if 'query' in col.lower() or 'text' in col.lower() or 'سؤال' in col][0]
domain_col = [col for col in df.columns if 'domain' in col.lower() or 'ministry' in col.lower() or 'جهة' in col][0]

print(f"Using '{query_col}' for user entries and '{domain_col}' for ministries.\n")

# Clean columns
df[query_col] = df[query_col].astype(str).fillna('')
df[domain_col] = df[domain_col].astype(str).fillna('Unknown/System Test')

# Word count
def CleanAndCountWords(text):
    return len([w for w in text.split() if w.strip()])

df['word_count'] = df[query_col].apply(CleanAndCountWords)

# Global metrics
total_records = len(df)
total_words = df['word_count'].sum()
avg_words = df['word_count'].mean()

print(f"--- Global Metrics ---")
print(f"Total Records: {total_records}")
print(f"Total Words: {total_words}")
print(f"Average Words: {avg_words:.2f}\n")

# =========================
# ✅ DATA CLEANING
# =========================

greetings = [
    "hi", "hello", "مرحبا", "السلام عليكم", "اهلا", "هلا",
    "good morning", "good evening", "حيالله"
]

def remove_greetings(text):
    text_lower = text.lower().strip()
    for g in greetings:
        if text_lower == g:
            return ""
    return text

df[query_col] = df[query_col].apply(remove_greetings)
df = df[df[query_col].str.strip() != ""]

# =========================
# ✅ TRANSLATION (Arabic → English)
# =========================

print(f"🌐 Translating {len(df)} queries to English...")
print("   (This may take several minutes — please wait)\n")

translations = []
for i, text in enumerate(df[query_col]):
    translations.append(translate_to_english(text))
    if (i + 1) % 100 == 0:
        print(f"   ✅ Translated {i + 1} / {len(df)} queries...")
    time.sleep(0.05)  # avoid hitting API rate limits

df["query_english"] = translations
print("✅ Translation complete!\n")

# =========================
# ✅ INTENT CLASSIFICATION
# =========================

def classify_intent(text):
    text = text.lower()
    if any(word in text for word in ["لخص", "تلخيص", "summary", "summarize"]):
        return "Summarization"
    elif any(word in text for word in ["ترجم", "translate"]):
        return "Translation"
    elif any(word in text for word in ["اكتب", "write", "خطاب", "email"]):
        return "Email/Content Generation"
    else:
        return "Q&A / General"

df["intent"] = df[query_col].apply(classify_intent)

# ✅ Create analysis tables
intent_by_ministry = pd.crosstab(df[domain_col], df["intent"])
main_usage = intent_by_ministry.idxmax(axis=1)

top_email_ministry = intent_by_ministry["Email/Content Generation"].idxmax()

# =========================
# ✅ KEYWORD ANALYSIS
# =========================

stop_words = {
    'من','في','على','الى','إلى','عن','ما','هو','هي','كم','هل','او','أو',
    'the','to','a','is','for','of','and','in','on','this','that','you','i',
    'مع','أن','التي','فيه','عليه'
}

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

df["clean_text"] = df[query_col].apply(clean_text)

all_words = " ".join(df["clean_text"]).split()
filtered_words = [w for w in all_words if w not in stop_words and not w.isdigit()]
word_freq = Counter(filtered_words)

print("\nTop Topics:")
print(word_freq.most_common(20))

# =========================
# ✅ MINISTRY STATS
# =========================

unique_ministries = df[domain_col].unique()
summary_data = []

output_excel_path = r"C:\Users\abalbalushi\Downloads\Mueen_Report_v2.xlsx"

with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:

    for ministry in unique_ministries:
        min_df = df[df[domain_col] == ministry]

        min_total_records = len(min_df)
        min_total_words = min_df['word_count'].sum()
        min_avg_words = min_df['word_count'].mean()
        min_share = (min_total_records / total_records) * 100

        all_min_text = " ".join(min_df[query_col])
        all_words = re.findall(r'\b\w+\b', all_min_text.lower())

        meaningful_words = [w for w in all_words if w not in stop_words and not w.isdigit()]
        word_counts = Counter(meaningful_words)
        top_5_words = word_counts.most_common(5)

        top_words_str = ", ".join([f"{w[0]} ({w[1]})" for w in top_5_words])

        summary_data.append({
            'Ministry': ministry,
            'Total Records': min_total_records,
            'Volume %': round(min_share, 2),
            'Total Words': min_total_words,
            'Avg Words': round(min_avg_words, 2),
            'Top Words': top_words_str
        })

        # Save each ministry sheet — now includes English translation
        min_df[[query_col, 'query_english', 'word_count', 'intent']].to_excel(
            writer, sheet_name=str(ministry)[:30], index=False
        )

    # ✅ Overall summary
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Dashboard', index=False)

    # ✅ Full bilingual dataset in one sheet
    df[[domain_col, query_col, 'query_english', 'word_count', 'intent']].to_excel(
        writer, sheet_name='All Queries (Bilingual)', index=False
    )

    # ✅ Intent sheets
    intent_by_ministry.to_excel(writer, sheet_name='Intent by Ministry')

    main_usage_df = main_usage.reset_index()
    main_usage_df.columns = ['Ministry', 'Main Usage']
    main_usage_df.to_excel(writer, sheet_name='Main Usage', index=False)

    intent_by_ministry.sort_values(
        by="Email/Content Generation", ascending=False
    ).to_excel(writer, sheet_name='Email Ranking')

print(f"\n✅ Report saved successfully: {output_excel_path}")

# =========================
# ✅ FINAL INSIGHTS PRINT
# =========================

print("\n=== Main Usage per Ministry ===")
print(main_usage.head(10))

print("\n✅ Top Email Usage Ministry:")
print(top_email_ministry)

print("\n=== Sample Insights ===")
for ministry in list(intent_by_ministry.index)[:10]:
    print(f"{ministry} mainly uses Mueen for {intent_by_ministry.loc[ministry].idxmax()}")
