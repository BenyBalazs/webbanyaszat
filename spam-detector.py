import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("./data/spam.csv", encoding="utf-8")

data = raw_data[["v1", "v2"]]

print(data)

class_column = data['v1']
sms_data = data['v2']

corpus = sms_data
cv = CountVectorizer()
X = cv.fit_transform(corpus)

print(cv.get_feature_names_out())
print(X.toarray())

X_train, X_test, y_train, y_test = train_test_split(X, class_column, test_size=0.30, random_state=42)

clf = MultinomialNB()

clf.fit(X_train, y_train)

print("Accuracy of Model", clf.score(X_test, y_test) * 100, "%")

vect = cv.transform([
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030",
    "You've won a prize, a gift card, or a coupon that you need to redeem.",
    "Beautiful weekend coming up. Wanna go out? Sophie gave me your number. Check out my profile here: bit.ly/profileGB12RD",
    "Free pizza for the weekend pls visit: total.scam.com/give-me-your-creditcard"]).toarray()

asd = clf.predict(vect)

print(asd)
