from pandas import read_csv as read 
file = read("iris.csv")
data = file.values

print(file.isnull().sum())
X = data[:,0:-1]
Y = data[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xtrain, ytrain)

print(LR.predict(xtest))
print(LR.score(xtest, ytest))