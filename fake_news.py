import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import gensim


data_path = '/kaggle/input/fake-news/news.csv'

df = pd.read_csv(data_path, sep = ',')
#=====================================================
df['news'] = df['title'] + df['text']
df['label'] = df['label'].replace({'FAKE': 0, 'REAL': 1})


nltk.download('stopwords')


stop_words = stopwords.words('english')
stop_words.extend(['one', 'will'])
def preprocess(text, join_back=True):
    result = []
    for token in gensim.utils.simple_preprocess(text):

        if (token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_words):
            result.append(token)
    if join_back:
        result = " ".join(result)
    return result


df['filered_text']=df['news'].apply(preprocess)


X_train, X_test, y_train, y_test = train_test_split(df['filered_text'], df['label'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)


y_pred = classifier.predict(X_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train_tfidf, y_train)


y_pred = classifier.predict(X_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier()


classifier.fit(X_train_tfidf, y_train)


y_pred = classifier.predict(X_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))