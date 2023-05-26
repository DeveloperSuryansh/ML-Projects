from pandas import read_csv as read 
file = read("diabetes.csv")
data = file.values
# print(file)
# print(file.isna().sum())

X = data[:,0:-1]
Y = data[:,-1]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(criterion="entropy")
RFC.fit(xtrain, ytrain)

LR = LogisticRegression()
LR.fit(xtrain,ytrain)

print(LR.predict(xtest))
print(LR.score(xtest, ytest))

print(RFC.predict(xtest))
print(RFC.score(xtest, ytest))
