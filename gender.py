from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv as read
data = read("names.csv").values

X,Y = data[:50,0],data[:50,1]
CV = CountVectorizer(ngram_range=(1,2))
Xvec = CV.fit_transform(X).toarray()
Xtr,Xte,Ytr,Yte = train_test_split(Xvec,Y,test_size=0.2)
MNB = MultinomialNB()
LE = LabelEncoder()
MNB.fit(Xtr, LE.fit_transform(Ytr))
print(MNB.predict(Xte))
print(Yte)