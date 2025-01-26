# Fake News Classification using NLP

This project demonstrates a machine learning pipeline to classify news articles as fake or genuine using Natural Language Processing (NLP) techniques. The process involves data preprocessing, feature extraction, and model building.

---

## Project Overview

### 1. Dataset
The project uses two datasets:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains genuine news articles.

Both datasets are preprocessed and merged into a single dataset with the following structure:
- `text`: The content of the news article.
- `genuineness`: A binary label where 0 represents fake news and 1 represents genuine news.

### 2. Preprocessing Steps
1. **Tokenization**: Splitting text into individual words using the `word_tokenize` function from the NLTK library.
2. **Stemming**: Reducing words to their root forms using the Snowball Stemmer.
3. **Stopword Removal**: Filtering out short words to clean the text further.
4. **Data Splitting**: Splitting the data into training and testing sets using an 80-20 split.

### 3. Feature Engineering
- **TF-IDF Vectorization**: Transforming text data into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) method with `TfidfVectorizer`. The vectorizer is configured with a `max_df` of 0.7 to ignore overly common terms.

### 4. Models Used
1. **Logistic Regression**:
   - A simple yet effective classifier for binary classification tasks.
   - Achieved an accuracy score of `scr1` (value will be printed during runtime).

2. **Passive Aggressive Classifier**:
   - Suitable for large-scale learning with streaming data.
   - Achieved an accuracy score of `scr2` (value will be printed during runtime).

---

## Prerequisites
Ensure the following libraries are installed before running the script:
- `pandas`
- `nltk`
- `scikit-learn`

Install these using:
```bash
pip install nltk pandas scikit-learn
```

---

## Usage Instructions
1. Download the `Fake.csv` and `True.csv` datasets and save them in the specified directory.
2. Update the file paths in the script to point to the datasets.
3. Run the script to:
   - Preprocess the data.
   - Train the models.
   - Evaluate the models and print the accuracy scores.

---

## Script Walkthrough
### Key Sections:
- **Data Loading**: Merging and labeling the datasets.
- **Data Preprocessing**: Cleaning and tokenizing the text.
- **Feature Extraction**: Generating numerical features with TF-IDF.
- **Model Training and Evaluation**: Using Logistic Regression and Passive Aggressive Classifier for classification.

### Outputs:
- Predicted labels for the test set.
- Accuracy scores for both models.

---

## Improvements and Future Work
- Implement additional preprocessing techniques (e.g., lemmatization).
- Test with additional machine learning models or deep learning architectures (e.g., RNNs, BERT).
- Perform hyperparameter tuning to improve model performance.
- Visualize the data and results for better insights.

---

## Acknowledgments
- NLTK library for text preprocessing.
- Scikit-learn for model implementation.
