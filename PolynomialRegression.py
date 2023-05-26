from pandas import read_csv as read
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

data = read("Salary.csv")
x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

print(x)
print(y)

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
print(x_poly)

reg = LinearRegression()
reg.fit(x_poly, y)
print(reg.predict(poly.fit_transform([[5.5]])))