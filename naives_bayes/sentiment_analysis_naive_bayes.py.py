import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# Load Dataset

data = pd.read_csv("../datasets/IMDB Dataset.csv")

# Sample Dataset

df = data.sample(n=15000, random_state=42).reset_index(drop=True)

train_df, test_df = train_test_split(
    df,
    test_size=5000,
    random_state=42,
    stratify=df["sentiment"],
)


# Text Preprocessing

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def clean_html(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, " ", text)


def to_lower(text):
    return text.lower()


def remove_special(text):
    cleaned = ""

    for char in text:
        if char.isalnum() or char == " ":
            cleaned += char
        else:
            cleaned += " "

    return cleaned


def remove_stopwords(text):
    words = []

    for word in text.split():
        if word not in stop_words:
            words.append(word)

    return " ".join(words)


def stem_words(text):
    stemmed = []

    for word in text.split():
        stemmed.append(ps.stem(word))

    return " ".join(stemmed)


def preprocess(text):
    text = clean_html(text)
    text = to_lower(text)
    text = remove_special(text)
    text = remove_stopwords(text)
    text = stem_words(text)

    return text




train_df["review"] = train_df["review"].apply(preprocess)

# Preprocess Testing Data

test_df["review"] = test_df["review"].apply(preprocess)

# Feature Extraction

cv = CountVectorizer(max_features=500)

x_train = cv.fit_transform(train_df["review"])
x_test = cv.transform(test_df["review"])

y_train = train_df["sentiment"]
y_test = test_df["sentiment"]

# Models

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Training

gnb.fit(x_train.toarray(), y_train)
mnb.fit(x_train, y_train)
bnb.fit(x_train, y_train)

# Prediction

y_pred_gnb = gnb.predict(x_test.toarray())
y_pred_mnb = mnb.predict(x_test)
y_pred_bnb = bnb.predict(x_test)

# Evaluation

print("GaussianNB Accuracy    :", accuracy_score(y_test, y_pred_gnb))
print("MultinomialNB Accuracy :", accuracy_score(y_test, y_pred_mnb))
print("BernoulliNB Accuracy   :", accuracy_score(y_test, y_pred_bnb))


