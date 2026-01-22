## Email Spam Classifier - Comprehensive Documentation

### Project Overview
The **Email Spam Classifier** is a machine learning project that uses Natural Language Processing (NLP) and a Naive Bayes classifier to automatically detect spam emails. The model is trained on the SMS Spam Collection dataset, which serves as a proxy for email spam classification.

---

### Table of Contents
1. Features
2. Technical Architecture
3. Dependencies
4. Installation & Setup
5. Project Structure
6. How It Works
7. Code Explanation
8. Model Performance
9. Usage Examples
10. Key Parameters
11. Troubleshooting

---

### Features
- ✅ **Automatic Text Preprocessing**: Removes punctuation, converts to lowercase, removes stopwords
- ✅ **TF-IDF Vectorization**: Converts text to numerical features using Term Frequency-Inverse Document Frequency
- ✅ **Multinomial Naive Bayes Classifier**: Fast and effective for text classification
- ✅ **Automatic Dataset Download**: Retrieves the SMS Spam Collection dataset from UCI ML Repository
- ✅ **Comprehensive Evaluation**: Provides accuracy score and detailed classification metrics
- ✅ **Reusable Prediction Function**: `classify_email()` function for classifying new emails

---

### Technical Architecture

```
Input Email
    ↓
Text Preprocessing (clean & tokenize)
    ↓
TF-IDF Vectorization (convert to numbers)
    ↓
Multinomial Naive Bayes Classifier
    ↓
Output: "spam" or "not spam"
```

---

### Dependencies
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library (vectorizer, classifier, metrics)
- **nltk**: Natural Language Toolkit for stopword removal
- **requests**: HTTP library for downloading datasets
- **zipfile**: Standard library for handling ZIP archives
- **string**: Standard library for punctuation handling

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

### Installation & Setup

1. **Clone/Download the project** to your local machine

2. **Install dependencies**:
   ```bash
   pip install pandas scikit-learn nltk requests
   ```

3. **Run the script**:
   ```bash
   python Email_Spam_Classifier.py
   ```

**Note**: The script automatically downloads the SMS Spam Collection dataset on first run. Ensure you have internet connectivity.

---

### Project Structure

```
Email_Spam_Classifier.py     # Main script
requirements.txt              # Project dependencies
data/
  └── SMSSpamCollection       # Dataset (downloaded automatically)
README.md                      # Documentation
```

---

### How It Works

#### 1. **Data Acquisition**
- Downloads the SMS Spam Collection dataset from UCI ML Repository
- Dataset contains ~5,600 labeled SMS messages (ham/spam)

#### 2. **Data Preparation**
- Reads CSV file with labels ('ham' or 'spam') and messages
- Converts labels to binary (0 = ham/not spam, 1 = spam)
- Applies text preprocessing

#### 3. **Text Preprocessing**
- **Remove punctuation**: Eliminates special characters
- **Lowercase conversion**: Normalizes text
- **Stopword removal**: Removes common words (the, a, is, etc.) that don't add value

#### 4. **Feature Extraction (Vectorization)**
- Converts text into numerical vectors using TF-IDF
- Maintains top 5,000 features (most relevant words)
- Creates sparse matrices for efficient computation

#### 5. **Model Training**
- Splits data: 80% training, 20% testing
- Trains Multinomial Naive Bayes classifier
- Calculates word probabilities for each class

#### 6. **Evaluation**
- Tests model on unseen data
- Computes accuracy, precision, recall, and F1-score

---

### Code Explanation

**Text Preprocessing Function**:
```python
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
```
Cleans raw text by removing noise while preserving meaningful content.

**Vectorization**:
```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
```
Converts preprocessed text into 5,000-dimensional feature vectors based on word importance.

**Classification Function**:
```python
def classify_email(email):
    email = preprocess_text(email)
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]
    return 'spam' if prediction == 1 else 'not spam'
```
Preprocesses new email, vectorizes it, and returns spam/not spam prediction.

---

### Model Performance

The script outputs:
- **Accuracy**: Percentage of correct predictions on test set
- **Classification Report**: 
  - **Precision**: Of emails predicted as spam, how many actually are spam?
  - **Recall**: Of all spam emails, how many did the model catch?
  - **F1-Score**: Harmonic mean of precision and recall

**Expected Performance**: 
- Typical accuracy: 95%+ on SMS Spam Collection dataset
- High precision and recall due to distinct spam patterns

---

### Usage Examples

**Run the complete pipeline**:
```bash
python Email_Spam_Classifier.py
```

**Output Example**:
```
Accuracy: 0.98
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.98      0.98       967
           1       0.95      0.97      0.96       150

    accuracy                           0.98      1117
   macro avg       0.97      0.98      0.97      1117
weighted avg       0.98      0.98      0.98      1117

Example email: "Congratulations! You've won a free iPhone. Click here to claim."
Classification: spam
```

**Classify custom emails in interactive mode**:
```python
# Add this to the script:
while True:
    user_email = input("Enter email (or 'quit' to exit): ")
    if user_email.lower() == 'quit':
        break
    print(f"Classification: {classify_email(user_email)}\n")
```

---

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `test_size` | 0.2 | 20% data for testing, 80% for training |
| `max_features` | 5000 | Number of top words to use as features |
| `random_state` | 42 | Seed for reproducibility |
| `Label Mapping` | 0/1 | 0 = ham (not spam), 1 = spam |

---

### Troubleshooting

| Issue | Solution |
|-------|----------|
| NLTK stopwords not found | Run: `python -c "import nltk; nltk.download('stopwords')"` |
| Dataset download fails | Check internet connection or manually download from https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip and place in data folder |
| Low accuracy | Try increasing `max_features` in TfidfVectorizer or experiment with other models (SVM, RandomForest) |
| Memory issues | Reduce `max_features` or process data in batches |
| FileNotFoundError for 'SMSSpamCollection' | Ensure the script ran successfully or manually extract the dataset |

---

### Advantages & Limitations

**Advantages**:
- Fast training and prediction
- Handles text data natively
- Works well with smaller datasets
- Easy to interpret results

**Limitations**:
- Uses SMS data instead of real email (may need retraining for email-specific spam)
- Doesn't capture context or semantics deeply
- May struggle with new spam patterns
- Performance depends on training data quality

---

### Future Enhancements

1. **Real Email Dataset**: Train on actual email data
2. **Advanced Models**: Try Deep Learning (LSTM, BERT) for better context understanding
3. **Feature Engineering**: Add sender reputation, email headers, domain checks
4. **Active Learning**: Retrain on misclassified emails
5. **Multi-class Classification**: Detect phishing, promotional, normal emails separately
