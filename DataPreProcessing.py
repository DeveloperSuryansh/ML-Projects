import numpy as np
from pandas import read_csv as read
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Reading and Loading Datasets
dataset = read("Country_data.csv")
x = dataset[["Country","Age","Salary"]].values
y = dataset[["Purchased"]].values

# Step 2: Finding Missing Values and Replacing it
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
x[:,1:3]  = imputer.fit_transform(X=x[:,1:3])
print(x)

# Step 3: Encoding Categorical Data
labelencoder = LabelEncoder()
x[:,0] = labelencoder.fit_transform(x[:,0])
print(x)

onehotenc = OneHotEncoder()
label_x = onehotenc.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()
print(label_x)

# Step 3 for Y
y[:,0] = labelencoder.fit_transform(y[:,0])
print(y)

# Step 4: Splitting DataSets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Step 5: Feature Scaling
scaler = StandardScaler()
scx = scaler.fit_transform(x_train)
scxtest = scaler.transform(x_test)
print(scx,scxtest,sep="\n\n",end='\n\n')

scy = scaler.fit_transform(y_train)
scytest = scaler.transform(y_test)
print(scy,scytest,sep="\n\n",end="\n\n")