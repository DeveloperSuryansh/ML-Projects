from pandas import read_csv as read
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = read("diabetes.csv").values

x = data[:,0:-1]
y = data[:,-1]

xtr,xte,ytr,yte = train_test_split(x,y,test_size = 0.2,random_state=1)
log = LogisticRegression()
log.fit(xtr, ytr)
print("Test Data:",xte)
print("Original Ans: ",yte)
print("Predicted:",log.predict(xte))
print("\n\nScore: ",log.score(xte, yte))
print("\n\n\n\n")

tree = DecisionTreeClassifier(criterion="gini")
tree.fit(xtr,ytr)
print("Test Data:",xte)
print("Original Ans: ",yte)
print("Predicted:",tree.predict(xte))
print("\n\nScore: ",tree.score(xte, yte))
print("\n\n\n\n")

clf = RandomForestClassifier(n_estimators=100,criterion="entropy")
clf.fit(xtr, ytr)
print("Test Data:",xte)
print("Original Ans: ",yte)
print("Predicted:",clf.predict(xte))
print("\n\nScore: ",clf.score(xte, yte))
print("\n\n\n\n")

GNB = GaussianNB()
GNB.fit(xtr, ytr)
print("Test Data:",xte)
print("Original Ans: ",yte)
print("Predicted:",GNB.predict(xte))
print("\n\nScore: ",GNB.score(xte, yte))
print("\n\n\n\n")

svc = SVC(kernel="rbf")
svc.fit(x,y)
print("Test Data:",xte)
print("Original Ans: ",yte)
print("Predicted:",svc.predict(xte))
print("\n\nScore: ",svc.score(xte, yte))
print("\n\n\n\n")
