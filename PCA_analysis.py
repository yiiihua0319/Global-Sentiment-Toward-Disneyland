import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ======================
# 0. 設定檔案/輸出資料夾
# ======================

INPUT_FILE = "DisneylandReviews_clean.csv"
OUTPUT_DIR = "pca_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Step 0: Loading data ===")
df = pd.read_csv(INPUT_FILE)

# 我們只拿有文字的觀測
df = df.dropna(subset=["Clean_Text"]).reset_index(drop=True)

print(f"Loaded {len(df)} reviews with cleaned text.")


# ======================
# 1. 文字向量化 (TF-IDF)
# ======================

print("=== Step 1: Building TF-IDF matrix ===")

# max_features 可以調大 500 -> 1000 -> 2000，如果你電腦跑得動就增加
vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words='english',   # 去掉英文常見無意義字 like "the", "and"
    ngram_range=(1, 2)      # 包含 unigram + bigram（ex: "long wait"）
)

X = vectorizer.fit_transform(df["Clean_Text"])
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF matrix shape:", X.shape)  # (評論數, 字彙數)


# ======================
# 2. PCA 降維到 2D
# ======================

print("=== Step 2: Running PCA ===")
pca = PCA(n_components=2, random_state=42)

# 注意：X 是 sparse matrix，要先轉 dense
X_dense = X.toarray()
X_pca = pca.fit_transform(X_dense)

print("Explained variance ratio (2 PCs):", pca.explained_variance_ratio_)
# 這告訴你 PC1 + PC2 解釋了多少資訊

# 把 PCA 結果合併回 df，存起來
df_pca = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Sentiment_Score": df["Sentiment_Score"],
    "Rating": df["Rating"],
    "Branch": df["Branch"]
})

df_pca.to_csv(os.path.join(OUTPUT_DIR, "reviews_pca_2d.csv"), index=False)
print("[OK] Saved reviews_pca_2d.csv")


# ======================
# 3. 視覺化：用情緒當顏色
# ======================

print("=== Step 3: Plotting PCA colored by sentiment ===")

plt.figure(figsize=(8,6))
sc = plt.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    c=df_pca["Sentiment_Score"],
    s=5,
    cmap='coolwarm'
)
plt.colorbar(sc, label="Sentiment Score (-1 neg ~ +1 pos)")
plt.title("PCA of Disneyland Reviews (colored by sentiment)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter_sentiment.png"))
plt.close()
print("[OK] Saved pca_scatter_sentiment.png")


# ======================
# 4. 視覺化：用樂園 Branch 當顏色
# ======================

print("=== Step 4: Plotting PCA colored by park ===")

# 我們給每個 branch 一個數字顏色
branch_to_id = {b:i for i,b in enumerate(df_pca["Branch"].unique())}
df_pca["Branch_id"] = df_pca["Branch"].map(branch_to_id)

plt.figure(figsize=(8,6))
sc2 = plt.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    c=df_pca["Branch_id"],
    s=5
)
plt.title("PCA of Disneyland Reviews (colored by park)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter_branch.png"))
plt.close()
print("[OK] Saved pca_scatter_branch.png")


# ======================
# 5. 解釋主成分：哪些詞影響了 PC1 / PC2
# ======================
# pca.components_ 形狀是 (2, max_features)
# 第 i 個主成分對應每個詞彙的 loading（權重）

print("=== Step 5: Inspecting top words for each principal component ===")

n_top_words = 15
pc_word_summary = []

for pc_i, component in enumerate(pca.components_):
    # 排序：最大的 loading 代表這個詞對該 PC 貢獻高
    top_idx = np.argsort(component)[-n_top_words:][::-1]
    top_terms = [feature_names[j] for j in top_idx]
    top_vals = [component[j] for j in top_idx]

    print(f"\nTop terms for PC{pc_i+1}:")
    for term, val in zip(top_terms, top_vals):
        print(f"{term:20s}  loading={val:.4f}")

    pc_df = pd.DataFrame({
        "PC": f"PC{pc_i+1}",
        "Term": top_terms,
        "Loading": top_vals
    })
    pc_word_summary.append(pc_df)

pc_word_summary = pd.concat(pc_word_summary, ignore_index=True)
pc_word_summary.to_csv(os.path.join(OUTPUT_DIR, "pca_top_terms.csv"), index=False)
print("[OK] Saved pca_top_terms.csv")

print("=== DONE ===")
print(f"All outputs saved in: {OUTPUT_DIR}")
