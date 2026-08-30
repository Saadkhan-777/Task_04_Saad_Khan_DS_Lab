# NLP Sentiment Classification Pipeline | DecodeLab Task 04

An end-to-end Natural Language Processing (NLP) sentiment analysis pipeline built on the NLTK `movie_reviews` corpus. This repository covers custom preprocessing, POS-guided lemmatization, negation preservation, N-Gram TF-IDF feature extraction, Multinomial Naive Bayes classification, and a live web application powered by Gradio.

---

## 📌 Key Features

* **Custom Negation Preservation**: Standard stopword removal often strips context. This pipeline preserves crucial negation tokens (`not`, `never`, `don't`, `no`, etc.) to prevent sentiment inversion.
* **POS-Guided Lemmatization**: Integrates Penn Treebank POS-tag mapping with NLTK's `WordNetLemmatizer` for contextual root-word reduction (accurately distinguishing verb vs. noun forms).
* **N-Gram TF-IDF Vectorization**: Configured with `ngram_range=(1, 2)` and `max_features=10000` to capture key compound phrases like *"not good"* or *"exceeded expectations"*.
* **Interactive Web Interface**: Deployed using Gradio for real-time sentiment testing and edge-case validation on unseen text inputs.

---

## 📊 Model Performance & Benchmarks

| Metric / Component | Implementation Details |
| :--- | :--- |
| **Dataset** | NLTK `movie_reviews` (2,000 balanced samples) |
| **Preprocessing** | Lowercasing, Regex Cleaning, POS Lemmatization, Custom Stopwords |
| **Vectorization** | TF-IDF Vectorizer (`ngram_range=(1, 2)`, `max_features=10000`) |
| **Model** | Multinomial Naive Bayes (alpha = 1.0) |
| **Train/Test Split** | 80/20 Stratified Split |
| **Accuracy** | **81.50%** |
| **Interface** | Gradio Web App |

---

## 📁 Repository Structure

├── task04_sentiment_analysis.ipynb   # Full data processing, model training, and evaluation

├── app.py                            # Gradio interface launch script

├── requirements.txt                  # Python dependencies

└── README.md                         # Project documentation

---

## ⚙️ Installation & Usage

Run the following commands in Git Bash:

git clone https://github.com/Saadkhan-777/Task_04_Saad_Khan_DS_Lab.git
cd Task_04_Saad_Khan_DS_Lab
pip install -r requirements.txt
python app.py

---

## 🚀 Future Enhancements

* **Model Exploration**: Benchmark against `LinearSVC` and `LogisticRegression` to compare classification boundaries.
* **Explicit Negation Tagging**: Implement `NOT_` prefix transformations for tokens following negation words.
* **Hyperparameter Tuning**: Optimize TF-IDF parameters (`sublinear_tf=True`, custom `min_df`) via grid search.

---

## 🔗 Connect & Links

* **Repository**: https://github.com/Saadkhan-777/Task_04_Saad_Khan_DS_Lab.git
* **LinkedIn**: https://linkedin.com/in/saad-khan-a50953370

*Completed as part of the Data Science Internship at **DecodeLab**.*
