from pandas import read_csv as read
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

data = read("diabetes.csv").values
x,y = data[:,0:-1],data[:,-1]
model = RandomForestClassifier(criterion="entropy")
model.fit(x,y)
dump(model,"Diabetes.model")