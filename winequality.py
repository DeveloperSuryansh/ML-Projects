from pandas import read_csv as read 
file = read("winequality-red.csv")

data = file.values

print(file.isnull().sum())

X = data[:,0:-1]
Y = data[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.05)

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(criterion="entropy")
RFC.fit(xtrain, ytrain)

print(RFC.predict(xtest))
print(RFC.score(xtest, ytest))