import pandas as pd
import matplotlib.pyplot as plt
import os

# ========= 0. 準備輸出資料夾 =========
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 1. 讀資料 =========
INPUT_FILE = "DisneylandReviews_clean.csv"

df = pd.read_csv(INPUT_FILE)

print("Loaded data shape:", df.shape)
print(df.head())

# ========= 2. 基本資料檢查 =========
# 欄位型態/缺值
with open(os.path.join(OUTPUT_DIR, "data_info.txt"), "w") as f:
    f.write("=== DATA INFO ===\n")
    f.write(str(df.info(buf=None)) + "\n\n")
    f.write("=== NULL COUNTS ===\n")
    f.write(str(df.isnull().sum()) + "\n")

print("[OK] Saved data_info.txt")

# ========= 3. 敘述統計 (整體) =========
summary = {
    "Total_Reviews": len(df),
    "Avg_Rating": df["Rating"].mean(),
    "Std_Rating": df["Rating"].std(),
    "Avg_Sentiment": df["Sentiment_Score"].mean(),
    "Std_Sentiment": df["Sentiment_Score"].std(),
    "Min_Sentiment": df["Sentiment_Score"].min(),
    "Max_Sentiment": df["Sentiment_Score"].max()
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)
print("[OK] Saved summary_stats.csv")
print(summary_df)

# ========= 4. 星級分佈 =========
rating_counts = df["Rating"].value_counts().sort_index()
rating_pct = (df["Rating"].value_counts(normalize=True).sort_index() * 100).round(1)

rating_dist_df = pd.DataFrame({
    "Rating": rating_counts.index,
    "Count": rating_counts.values,
    "Percent": rating_pct.values
})
rating_dist_df.to_csv(os.path.join(OUTPUT_DIR, "rating_distribution.csv"), index=False)
print("[OK] Saved rating_distribution.csv")
print(rating_dist_df)

# 圖：星級分佈長條圖
plt.figure()
rating_counts.plot(kind="bar")
plt.title("Distribution of Ratings")
plt.xlabel("Rating (Stars)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"))
plt.close()
print("[OK] Saved rating_distribution.png")

# ========= 5. 情緒分數基本統計 =========
sentiment_desc = df["Sentiment_Score"].describe()
with open(os.path.join(OUTPUT_DIR, "sentiment_stats.txt"), "w") as f:
    f.write(str(sentiment_desc))

print("[OK] Saved sentiment_stats.txt")
print(sentiment_desc)

# 圖：情緒分數直方圖
plt.figure()
df["Sentiment_Score"].plot(kind="hist", bins=30)
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment_Score (-1 = negative, +1 = positive)")
plt.ylabel("Count of Reviews")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_hist.png"))
plt.close()
print("[OK] Saved sentiment_hist.png")

# ========= 6. 各樂園比較 (Branch) =========
branch_stats = df.groupby("Branch")[["Rating", "Sentiment_Score"]].mean().round(3)
branch_stats.to_csv(os.path.join(OUTPUT_DIR, "branch_stats.csv"))
print("[OK] Saved branch_stats.csv")
print(branch_stats)

# 圖：各樂園平均情緒分數
plt.figure()
branch_stats["Sentiment_Score"].plot(kind="bar")
plt.title("Average Sentiment by Park")
plt.xlabel("Park / Branch")
plt.ylabel("Avg Sentiment Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_sentiment_by_branch.png"))
plt.close()
print("[OK] Saved avg_sentiment_by_branch.png")

# 圖：各樂園平均星級
plt.figure()
branch_stats["Rating"].plot(kind="bar")
plt.title("Average Rating by Park")
plt.xlabel("Park / Branch")
plt.ylabel("Avg Star Rating")
plt.ylim(0,5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_rating_by_branch.png"))
plt.close()
print("[OK] Saved avg_rating_by_branch.png")

# ========= 7. 時間趨勢 (Year_Month) =========
# 轉成 datetime / period 型別（保險一下）
if "Year_Month" in df.columns:
    # 嘗試處理成可以排序的時間欄
    df["Year_Month"] = pd.to_datetime(df["Year_Month"], errors="coerce")

    monthly_stats = (
        df.groupby("Year_Month")[["Rating", "Sentiment_Score"]]
          .mean()
          .sort_index()
          .round(3)
    )

    monthly_stats.to_csv(os.path.join(OUTPUT_DIR, "monthly_stats.csv"))
    print("[OK] Saved monthly_stats.csv")
    print(monthly_stats.head())

    # 圖：每月平均情緒
    plt.figure()
    monthly_stats["Sentiment_Score"].plot()
    plt.title("Monthly Average Sentiment")
    plt.xlabel("Month")
    plt.ylabel("Avg Sentiment Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_sentiment_trend.png"))
    plt.close()
    print("[OK] Saved monthly_sentiment_trend.png")

    # 圖：每月平均星級
    plt.figure()
    monthly_stats["Rating"].plot()
    plt.title("Monthly Average Rating")
    plt.xlabel("Month")
    plt.ylabel("Avg Rating")
    plt.ylim(0,5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_rating_trend.png"))
    plt.close()
    print("[OK] Saved monthly_rating_trend.png")

else:
    print("⚠ 沒有 Year_Month 欄位，跳過時間趨勢分析。")

# ========= 8. Rating vs Sentiment 的相關性 =========
corr_value = df["Rating"].corr(df["Sentiment_Score"])
with open(os.path.join(OUTPUT_DIR, "correlation.txt"), "w") as f:
    f.write(f"Correlation between Rating and Sentiment_Score: {corr_value}\n")

print("[OK] Saved correlation.txt")
print("Correlation between Rating and Sentiment_Score:", corr_value)

print("✅ Done. All outputs are in the 'analysis_output' folder.")
