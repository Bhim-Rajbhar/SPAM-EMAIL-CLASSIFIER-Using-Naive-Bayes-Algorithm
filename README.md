# 📧 Spam Email Classifier Using Naive Bayes Algorithm

A machine learning project that automatically detects whether an email is **spam** or **ham (not spam)** using the Naive Bayes classification algorithm. This was built as part of my ML learning journey during my MCA program with an AI/ML specialization.

---

## 🔍 What This Project Does

Ever wondered how your email app knows to send certain emails straight to the spam folder? This project does exactly that — it reads the content of an email and predicts whether it's spam or a legitimate message.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Scikit-learn | Model building & evaluation |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| NLTK | Text preprocessing |
| Matplotlib / Seaborn | Data visualization |

---

## 📁 Project Structure

```
SPAM-EMAIL-CLASSIFIER/
│
├── dataset/
│   └── spam.csv               # Email dataset (SMS Spam Collection)
│
├── spam_classifier.py         # Main Python script
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation
```

---

## ⚙️ How It Works

1. **Load the Dataset** — Used the popular SMS Spam Collection dataset
2. **Preprocess the Text** — Removed stopwords, punctuation, and applied lowercase conversion
3. **Feature Extraction** — Converted text to numbers using TF-IDF Vectorizer
4. **Train the Model** — Applied Multinomial Naive Bayes classifier
5. **Evaluate the Model** — Checked accuracy, precision, recall, and confusion matrix

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision | ~96% |
| Recall | ~94% |

> Results may vary slightly depending on train/test split.

---

## 🚀 How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/Bhim-Rajbhar/SPAM-EMAIL-CLASSIFIER-Using-Naive-Bayes-Algorithm.git
cd SPAM-EMAIL-CLASSIFIER-Using-Naive-Bayes-Algorithm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python spam_classifier.py
```

---

## 📌 What I Learned

- How to clean and preprocess real-world text data
- How TF-IDF works and why it's useful for NLP tasks
- How Naive Bayes works for text classification
- How to evaluate a classification model properly

---

## 🙋‍♂️ About Me

**Bhim Rajbhar**  
MCA Student | AI/ML Specialization  
Passionate about building real-world ML projects and growing toward a career as a Machine Learning Engineer.

---

## 📃 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful or interesting, feel free to star the repository!
