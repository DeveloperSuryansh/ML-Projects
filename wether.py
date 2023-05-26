from pickle import load
file = open("Weather_model.model","rb")
a = load(file)
print(a.predict([[1,35,23,4]]))