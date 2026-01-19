import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("IMDB_Dataset.csv")
df = df[['review', 'sentiment']]

print("Dataset Shape:", df.shape)
print(df['sentiment'].value_counts())

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

print("Cleaning text data... (may take a few minutes)")
df['clean_review'] = df['review'].apply(preprocess_text)
print("Cleaning completed.")

positive_text = " ".join(df[df['sentiment'] == 'positive']['clean_review'])
negative_text = " ".join(df[df['sentiment'] == 'negative']['clean_review'])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
wc_pos = WordCloud(width=600, height=400).generate(positive_text)
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews WordCloud')

plt.subplot(1,2,2)
wc_neg = WordCloud(width=600, height=400).generate(negative_text)
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews WordCloud')
plt.show()

X = df['clean_review']
y = df['sentiment'].map({'negative': 0, 'positive': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

pipelines = {
    'LogisticRegression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'MultinomialNB': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LinearSVC(random_state=42))
    ])
}

results = {}
for name, model in pipelines.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {'Accuracy': acc, 'F1': f1}
    print(f"{name} - Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

best_model = pipelines['LogisticRegression']
y_pred = best_model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative','Positive']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

vect = best_model.named_steps['tfidf']
clf = best_model.named_steps['clf']
feature_names = np.array(vect.get_feature_names_out())

if hasattr(clf, 'coef_'):
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]
    print("\nTop Positive Words:", feature_names[top_pos_idx])
    print("\nTop Negative Words:", feature_names[top_neg_idx])

param_grid = {
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]
}

grid = GridSearchCV(pipelines['LogisticRegression'], param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
best_tuned_model = grid.best_estimator_

y_pred_tuned = best_tuned_model.predict(X_test)
print("\nTuned Model F1 Score:", f1_score(y_test, y_pred_tuned))

joblib.dump(best_tuned_model, "sentiment_model.pkl")
print("\nModel saved as sentiment_model.pkl")

def predict_sentiment(review_text):
    clean_text = preprocess_text(review_text)
    prediction = best_tuned_model.predict([clean_text])[0]
    return "Positive" if prediction == 1 else "Negative"

test_reviews = [
    "This movie was absolutely amazing. I loved every part of it!",
    "Terrible movie. Waste of time and money!"
]

for review in test_reviews:
    sentiment = predict_sentiment(review)
    print(f"\nReview: {review}\nPredicted Sentiment: {sentiment}")
