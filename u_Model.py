import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import pickle


# 1. LOAD DATA

df = pd.read_csv("clean_data.csv")

df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["label"] = df["label"].astype(int)



# 2. BALANCE DATASET (IMPORTANT FIX)

df = shuffle(df).reset_index(drop=True)

real = df[df["label"] == 1]   # REAL news
fake = df[df["label"] == 0]   # FAKE news

min_len = min(len(real), len(fake))

df_balanced = pd.concat([real[:min_len], fake[:min_len]])

df = shuffle(df_balanced).reset_index(drop=True)

print("Balanced dataset size:", df.shape)
print(df['label'].value_counts())

# 3. TRAIN–TEST SPLIT

x = df["clean_text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)



# 4. PIPELINE WITH FIXED TF-IDF

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),          
        max_df=0.9,
        min_df=2,                    
        sublinear_tf=True
    )),
    ("pac", PassiveAggressiveClassifier())
])



# 5. GRID SEARCH TO IMPROVE PAC MODEL

param_grid = {
    "pac__max_iter": [500, 1000],
    "pac__C": [0.5, 1.0],
    "pac__loss": ["hinge", "squared_hinge"]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)



# 6. TRAIN THE MODEL

print("\nTraining model...")
grid.fit(x_train, y_train)

print("\nBest Parameters:", grid.best_params_)



# 7. EVALUATE PERFORMANCE

y_pred = grid.predict(x_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# 8. SAVE MODEL + VECTORIZER

best_model = grid.best_estimator_

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(best_model.named_steps["tfidf"], open("tfidf.pkl", "wb"))

print("\nModel and vectorizer saved.")



# 9. SAVE EXPLAINABILITY (WORD WEIGHTS)

feature_names = best_model.named_steps["tfidf"].get_feature_names_out()
weights = best_model.named_steps["pac"].coef_[0]

explain_dict = {word: float(weights[i]) for i, word in enumerate(feature_names)}

pickle.dump(explain_dict, open("explain_words.pkl", "wb"))


print("Explainability saved.")
