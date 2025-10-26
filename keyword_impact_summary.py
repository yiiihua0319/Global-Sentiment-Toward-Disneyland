import pandas as pd
import os
import re
import numpy as np

# ======================
# 0. 設定檔案與輸出路徑
# ======================
INPUT_FILE = "DisneylandReviews_clean.csv"
OUTPUT_DIR = "keyword_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Step 0: Loading data ===")
df = pd.read_csv(INPUT_FILE)

# 確保文字欄位存在
if "Clean_Text" not in df.columns:
    raise ValueError("Clean_Text column not found. Make sure you ran the cleaning step first.")

# ======================
# 1. 定義主題關鍵字群
#    這些就像你在長榮專案會挑“延誤”“服務”“餐點”
#    這裡我們挑的是遊樂園體驗常見痛點/亮點
# ======================
keyword_groups = {
    "wait_time": [
        r"wait", r"waiting", r"queue", r"line", r"long line", r"line was long",
        r"queued", r"hour line", r"1 hour", r"2 hours"
    ],
    "price_value": [
        r"expensive", r"price", r"overpriced", r"too expensive", r"rip ?off",
        r"cost a lot", r"not worth", r"money worth"
    ],
    "staff_service": [
        r"staff", r"crew", r"employee", r"cast member", r"rude", r"helpful",
        r"friendly", r"kind", r"service", r"attitude"
    ],
    "food_quality": [
        r"food", r"meal", r"burger", r"fries", r"lunch", r"dinner", r"snack",
        r"restaurant", r"tasty", r"delicious", r"bad food"
    ],
    "attractions_fun": [
        r"ride", r"rides", r"roller coaster", r"attraction", r"show", r"fireworks",
        r"parade", r"fun", r"amazing", r"awesome", r"best day"
    ],
    "magic_experience": [
        r"magical", r"magic", r"dream", r"dreams come true", r"happiest place",
        r"love disney", r"disney magic", r"wonderful", r"unforgettable"
    ]
}

# ======================
# 2. 根據每組關鍵字，建立 0/1 標記欄位
# ======================

def build_flag_column(text_series, pattern_list):
    """
    對於一個文字欄位 (Series)，回傳一個 0/1 Series，
    1 表示該評論有出現 pattern_list 裡任一關鍵字（不分大小寫）
    """
    combined_pattern = r"(" + r"|".join(pattern_list) + r")"
    # re.IGNORECASE -> 不分大小寫
    return text_series.str.contains(combined_pattern, flags=re.IGNORECASE, regex=True).astype(int)

print("=== Step 1: Creating keyword flags ===")
for group_name, patterns in keyword_groups.items():
    col_name = f"has_{group_name}"
    df[col_name] = build_flag_column(df["Clean_Text"], patterns)

flag_cols = [c for c in df.columns if c.startswith("has_")]

# ======================
# 3. 計算每個主題 vs. 滿意度 指標
#    我們做兩件事：
#    (A) 這個主題被提到的比例
#    (B) 提到 vs. 沒提到 → 平均 Rating / Sentiment 的差
# ======================

print("=== Step 2: Aggregating impact stats ===")

rows = []
for col in flag_cols:
    group_name = col.replace("has_", "")

    mentioned = df[df[col] == 1]
    not_mentioned = df[df[col] == 0]

    row = {
        "topic": group_name,
        "mention_rate_%": round(df[col].mean() * 100, 2),
        "avg_rating_if_mentioned": round(mentioned["Rating"].mean(), 3) if len(mentioned) > 0 else np.nan,
        "avg_rating_if_not_mentioned": round(not_mentioned["Rating"].mean(), 3) if len(not_mentioned) > 0 else np.nan,
        "avg_sentiment_if_mentioned": round(mentioned["Sentiment_Score"].mean(), 3) if len(mentioned) > 0 else np.nan,
        "avg_sentiment_if_not_mentioned": round(not_mentioned["Sentiment_Score"].mean(), 3) if len(not_mentioned) > 0 else np.nan,
        "rating_diff": round(
            (mentioned["Rating"].mean() - not_mentioned["Rating"].mean()), 3
        ) if len(mentioned) > 0 and len(not_mentioned) > 0 else np.nan,
        "sentiment_diff": round(
            (mentioned["Sentiment_Score"].mean() - not_mentioned["Sentiment_Score"].mean()), 3
        ) if len(mentioned) > 0 and len(not_mentioned) > 0 else np.nan,
        "n_reviews_mentioned": len(mentioned),
        "n_reviews_not_mentioned": len(not_mentioned)
    }

    rows.append(row)

impact_df = pd.DataFrame(rows)

# sort by哪些主題影響評分最負面 (最小 rating_diff)
impact_df = impact_df.sort_values(by="rating_diff")

print("=== Step 3: Saving summary table ===")
impact_path = os.path.join(OUTPUT_DIR, "keyword_impact_summary.csv")
impact_df.to_csv(impact_path, index=False)
print("[OK] Saved:", impact_path)

print("\n=== Keyword Impact Summary (preview) ===")
print(impact_df[[
    "topic",
    "mention_rate_%",
    "avg_rating_if_mentioned",
    "avg_rating_if_not_mentioned",
    "rating_diff",
    "avg_sentiment_if_mentioned",
    "avg_sentiment_if_not_mentioned",
    "sentiment_diff",
    "n_reviews_mentioned"
]].to_string(index=False))

# ======================
# 4. 依樂園拆開看（跨園區比較）
#    例如：在 Paris，"wait_time" 出現時的平均評分是不是掉更多？
# ======================

print("\n=== Step 4: Breakdown by Branch ===")
branch_rows = []

for branch, sub in df.groupby("Branch"):
    for col in flag_cols:
        group_name = col.replace("has_", "")
        mentioned = sub[sub[col] == 1]
        not_mentioned = sub[sub[col] == 0]

        branch_rows.append({
            "branch": branch,
            "topic": group_name,
            "mention_rate_%": round(sub[col].mean() * 100, 2),
            "avg_rating_if_mentioned": round(mentioned["Rating"].mean(), 3) if len(mentioned) > 0 else np.nan,
            "avg_rating_if_not_mentioned": round(not_mentioned["Rating"].mean(), 3) if len(not_mentioned) > 0 else np.nan,
            "rating_diff": round(
                (mentioned["Rating"].mean() - not_mentioned["Rating"].mean()), 3
            ) if len(mentioned) > 0 and len(not_mentioned) > 0 else np.nan,
            "avg_sentiment_if_mentioned": round(mentioned["Sentiment_Score"].mean(), 3) if len(mentioned) > 0 else np.nan,
            "avg_sentiment_if_not_mentioned": round(not_mentioned["Sentiment_Score"].mean(), 3) if len(not_mentioned) > 0 else np.nan,
            "sentiment_diff": round(
                (mentioned["Sentiment_Score"].mean() - not_mentioned["Sentiment_Score"].mean()), 3
            ) if len(mentioned) > 0 and len(not_mentioned) > 0 else np.nan,
            "n_reviews_mentioned": len(mentioned),
            "n_reviews_not_mentioned": len(not_mentioned)
        })

branch_impact_df = pd.DataFrame(branch_rows)

branch_impact_path = os.path.join(OUTPUT_DIR, "keyword_impact_by_branch.csv")
branch_impact_df.to_csv(branch_impact_path, index=False)
print("[OK] Saved:", branch_impact_path)

print("\n=== Keyword Impact by Branch (preview) ===")
print(branch_impact_df.head(20).to_string(index=False))

print("\n=== DONE ===")
