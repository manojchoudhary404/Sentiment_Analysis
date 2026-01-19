# 🎬 Sentiment Analysis on Movie Reviews

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/) [![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)](#) [![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-purple)](#) [![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)](https://scikit-learn.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/manojchoudhary404)

<img width="1200" height="565" alt="Screenshot 2026-01-19 at 11 56 17 AM" src="https://github.com/user-attachments/assets/e5853c94-c21e-4740-913d-7b3c83847935" />
<img width="642" height="544" alt="Screenshot 2026-01-19 at 11 56 46 AM" src="https://github.com/user-attachments/assets/1c20403a-6a53-48dc-b729-7a63346db360" />

## 🔹 Overview
A **Machine Learning–based Sentiment Analysis System** that classifies **movie reviews** as **Positive** or **Negative** using **Natural Language Processing (NLP)** techniques and supervised learning algorithms. This project demonstrates **data cleaning, text preprocessing, feature extraction, model training, evaluation, and prediction** on real-world textual data.

## 🚀 Features
- 📝 Text preprocessing (HTML removal, punctuation cleaning, stopword removal, lemmatization)
- 📊 TF-IDF feature extraction
- 🤖 Multiple ML models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- 📈 Model evaluation using Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ☁️ WordCloud visualization for positive & negative reviews
- 💾 Model persistence using `joblib`
- 🔮 Predict sentiment for custom user input

## 🛠️ Tech Stack
- **Python 3.x**
- **Pandas & NumPy** – Data handling
- **NLTK** – Text preprocessing
- **Scikit-learn** – Machine Learning models
- **Matplotlib & Seaborn** – Visualization
- **WordCloud** – Text visualization

## 📂 Dataset
- **Source:** Kaggle
- **Dataset Name:** IMDb Movie Reviews Dataset
- **Labels:** positive, negative
> The dataset contains raw reviews including HTML tags, emojis, and special characters, making it suitable for real-world NLP preprocessing.

## 📦 Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/manojchoudhary404/Sentiment-Analysis.git
cd Sentiment-Analysis
```
2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
3. Install dependencies:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud joblib
```

## ▶️ Running the Project
Place `IMDB_Dataset.csv` in the project directory and run:
```bash
python3 sentiment.py
```
Output includes dataset statistics, model evaluation metrics, confusion matrix visualizations, WordClouds for positive & negative reviews, and sentiment predictions for sample reviews.

## 📂 Project Structure
```
Sentiment-Analysis/
│── IMDB_Dataset.csv        # Dataset
│── sentiment.py            # Main Python script
│── sentiment_model.pkl     # Saved trained model
│── README.md               # Project documentation
└── venv/                   # Virtual environment (ignored in git)
```

## 📊 Algorithms Used
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

## 🎯 Learning Outcomes
- Understanding text preprocessing in NLP
- Applying ML algorithms to text classification
- Evaluating classification models
- Handling small and imbalanced datasets
- Building an end-to-end ML project

## 🔮 Future Enhancements
- Deploy as a web app using Flask / FastAPI
- Add real-time review input via UI
- Integrate deep learning models (LSTM, BERT)
- Improve performance with larger datasets

## 📜 License
This project is licensed under the MIT License. © 2026 Manoj Choudhary

## 👨‍💻 Author
Manoj Choudhary | 🎓 MCA Student | Backend Devel![Uploading Screenshot 2026-01-19 at 11.56.17 AM.png…]()
oper | NLP & Machine Learning Enthusiast | GitHub: [https://github.com/manojchoudhary404](https://github.com/manojchoudhary404) | LinkedIn: [https://www.linkedin.com/in/manojchoudhary](https://www.linkedin.com/in/manojchoudhary) | Email: manojchoudhary.7.in@gmail.com

