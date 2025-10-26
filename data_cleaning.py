import pandas as pd
from textblob import TextBlob
import re

# read data
df = pd.read_csv("DisneylandReviews.csv", encoding="latin-1")

# cleanin duplicate and missing values
df = df.drop_duplicates(subset=["Review_Text"])
df = df.dropna(subset=["Review_Text"])

# formatting date column
df["Year_Month"] = pd.to_datetime(df["Year_Month"], errors="coerce").dt.to_period("M")

# cleaning text data
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)  # remove URLs and HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special characters
    return text.strip()

df["Clean_Text"] = df["Review_Text"].apply(clean_text)

# sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df["Sentiment_Score"] = df["Clean_Text"].apply(get_sentiment)

# saving cleaned data
df.to_csv("DisneylandReviews_clean.csv", index=False)
print(df.head())
