import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load data
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

# Lowercase
df['clean_text'] = df['text'].astype(str).str.lower()

# Remove URLs
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', " ", text)
df['clean_text'] = df['clean_text'].apply(remove_url)

# Remove HTML tags
def remove_html_tags(text):
    return re.sub(r'<.*?>', ' ', text)
df['clean_text'] = df['clean_text'].apply(remove_html_tags)

# Remove punctuation
def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))
df['clean_text'] = df['clean_text'].apply(remove_punctuations)

# Remove special characters
def remove_spl_chars(text):
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
df['clean_text'] = df['clean_text'].apply(remove_spl_chars)

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])
df['clean_text'] = df['clean_text'].apply(remove_stopwords)

# FAST lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def simple_lemma(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df['clean_text'] = df['clean_text'].apply(simple_lemma)

df.to_csv("clean_data.csv", index=False)
print("clean_data.csv saved successfully!")

