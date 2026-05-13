
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# ── 1. Load Dataset ──────────────────────────────────────────────
# Use SpamAssassin CSV (easy to download) or Enron processed CSV
# Columns: 'label' (spam/ham), 'text'
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print(df['label'].value_counts())

# ── 2. Text Preprocessing ────────────────────────────────────────
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text)       # Remove punctuation/numbers
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

# ── 3. Feature Extraction ─────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# ── 4. Train/Test Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 5. Train Naive Bayes ──────────────────────────────────────────
model = MultinomialNB()
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ── 7. Confusion Matrix ───────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Naive Bayes Spam Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ── 8. Predict Custom Email ───────────────────────────────────────
def predict_email(email_text):
    cleaned = preprocess(email_text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "🚨 SPAM" if result == 1 else "✅ HAM"
    print(f"\nEmail: {email_text[:60]}...")
    print(f"Prediction: {label}")
    print(f"Confidence → Ham: {prob[0]:.2%} | Spam: {prob[1]:.2%}")

predict_email("Congratulations! You've won a $1000 gift card. Click here now!")
predict_email("Hi team, please find the meeting notes attached.")