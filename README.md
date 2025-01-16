Fake News Classification

This project demonstrates a Fake News Classification pipeline using Natural Language Processing (NLP) techniques and Machine Learning models to distinguish between fake and genuine news articles.

Table of Contents

Installation

Dataset

Data Preprocessing

Tokenization

Stemming

Removing Stop Words

Splitting Dataset

Vectorization (TF-IDF)

Machine Learning Models

Logistic Regression

Passive Aggressive Classifier

Results

Installation

Clone the repository:

git clone <repository_url>

Navigate to the project directory and install dependencies:

pip install nltk pandas scikit-learn

Download the NLTK 'punkt' tokenizer:

import nltk
nltk.download('punkt')

Dataset

Fake.csv: Contains fake news articles.

True.csv: Contains genuine news articles.

Dataset Preparation

Both datasets are combined, and a new column genuineness is added:

0: Fake news

1: Genuine news

Columns like title, subject, and date are dropped to focus on the text content.

Data Preprocessing

Tokenization

The text column is tokenized into individual words using NLTK's word_tokenize function.

Stemming

Words are reduced to their root form using the Snowball Stemmer from NLTK.

Removing Stop Words

Words with a length less than or equal to 2 are removed.

Example Code:

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

data['text'] = data['text'].apply(word_tokenize)

sb = SnowballStemmer("english", ignore_stopwords=False)

def stem_it(text):
  return [sb.stem(word) for word in text]

data['text'] = data['text'].apply(stem_it)

def stopword_remover(text):
  return [word for word in text if len(word) > 2]

data['text'] = data['text'].apply(stopword_remover)
data['text'] = data['text'].apply(' '.join)

Splitting Dataset

The dataset is split into training and testing sets:

Training set: 75%

Testing set: 25%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['genuineness'], test_size=0.25)

Vectorization (TF-IDF)

The text data is vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) method:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

Machine Learning Models

Logistic Regression

A Logistic Regression model is trained with a maximum of 900 iterations.

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(max_iter=900)
model1.fit(tfidf_train, y_train)

pred1 = model1.predict(tfidf_test)
from sklearn.metrics import accuracy_score
scr1 = accuracy_score(y_test, pred1)

Passive Aggressive Classifier

A Passive Aggressive Classifier is trained to handle binary classification tasks.

from sklearn.linear_model import PassiveAggressiveClassifier

model2 = PassiveAggressiveClassifier(max_iter=100)
model2.fit(tfidf_train, y_train)

pred2 = model2.predict(tfidf_test)
scr2 = accuracy_score(y_test, pred2)

Results

Model

Accuracy

Logistic Regression

scr1

Passive Aggressive Classifier

scr2

Contributing

Feel free to open issues or submit pull requests for improvements or new features!

