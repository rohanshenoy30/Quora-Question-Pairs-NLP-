
# Quora Question Pairs NLP

## Project Overview

This project tackles the **Quora Question Pairs** Kaggle competition, where the goal is to determine if two questions are semantically similar. Identifying duplicate or similar questions helps reduce redundancy on the platform by merging answers under a single question, improving user experience and data quality.

**Problem Statement:**
On Quora, similar questions often appear as separate entries, leading to fragmented answers across different sections. This project uses Natural Language Processing (NLP) techniques to detect whether two given questions are duplicates (semantically similar). By accurately identifying duplicates, we can consolidate questions and their answers into one unified section.

---

## Dataset

The dataset consists of question pairs labeled as duplicates or not. Each row contains two questions and a label indicating whether they are semantically similar.

* `question1`: First question text
* `question2`: Second question text
* `is_duplicate`: Label (1 if questions are duplicates, else 0)

---

## Approach

1. **Data Preprocessing**

   * Text cleaning (lowercasing, removing punctuation)
   * Tokenization and stopword removal

2. **Feature Engineering**
   Extract meaningful features from question pairs including:

   * Length-based features (difference in token count)
   * Token overlap features (common words, stop words)
   * Longest common substring length
   * Fuzzy matching scores (QRatio, Partial Ratio, etc.)
   * Bag-of-Words (BoW) vectors using `CountVectorizer`

3. **Modeling**

   * Train a machine learning classifier (Random Forest) on extracted features.
   * Tune hyperparameters for better accuracy.

4. **Evaluation**

   * Measure performance with accuracy, precision, recall, and F1-score.
   * Use cross-validation on training data.

---

## How to Use

1. **Train the model** using the provided dataset and feature extraction pipeline.
2. **Save** the trained model and vectorizer for future predictions.
3. **Load** the model and vectorizer to predict if any new question pair is duplicate.
4. **Input** two questions into the prediction function to get a similarity score or binary duplicate prediction.

---

## Example

```python
q1 = 'Where is the capital of India?'
q2 = 'What is the current capital of Pakistan?'

# Load model and vectorizer
rf = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

# Prepare features
features = query_point_creator(q1, q2)  # Feature extraction function

# Predict similarity
prediction = rf.predict(features)
print("Are questions duplicates?", prediction[0])
```

---

## Dependencies

* Python 3.x
* numpy
* pandas
* scikit-learn
* nltk
* fuzzywuzzy
* pickle
* difflib

Make sure to install necessary packages, for example:

```bash
pip install numpy pandas scikit-learn nltk fuzzywuzzy
```

---

## Future Improvements

* Experiment with deep learning models (e.g., Siamese networks, BERT embeddings).
* Include more advanced text preprocessing (lemmatization, spelling correction).
* Use more sophisticated vectorizers like TF-IDF or word embeddings.
* Hyperparameter tuning with GridSearch or RandomSearch.
* Deployment as an API for real-time duplicate question detection.

---

## License

This project is open-source and free to use for educational and research purposes.

---
