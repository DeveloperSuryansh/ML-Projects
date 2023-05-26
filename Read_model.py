from pandas import read_csv
from joblib import load
from sklearn.model_selection import train_test_split
data = read_csv("diabetes.csv").values
model = load("Diabetes.model")
xtr,xte,ytr,yte = train_test_split(data[:,0:-1],data[:,-1],test_size=0.1)
print(model.predict(xte))