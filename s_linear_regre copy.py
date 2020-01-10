#simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# splitting the data set into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3,random_state=0)


#feature scaling

"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results

y_pred=regressor.predict(X_test)

#visualising the training set results

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience graph(training set)')
plt.Xtable('years of experience')
plt.ytable('salary')
plt.show()


   




