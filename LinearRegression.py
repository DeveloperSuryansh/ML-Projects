import pandas as pd
from sklearn import linear_model

dataset = pd.read_csv("InsurancePremium.csv")
linear = linear_model.LinearRegression()
linear.fit(dataset[["Age"]], dataset[["Premium"]])
print("Dataset =>\n",dataset,end="\n\n")
print("Prediction of Premiums for 21 Ages =>",linear.predict([[21]]))
print("Prediction of Premiums for 50 Ages =>",linear.predict([[50]]))