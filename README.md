# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Algorithm for Linear Regression Model

1. Import Libraries: Import numpy, pandas, and StandardScaler from sklearn.preprocessing.

2. Read Data: Read '50_Startups.csv' into a DataFrame (data) using pd.read_csv().

3. Data Preparation:
   - Extract features (X) and target variable (y) from the DataFrame.
   - Convert features to a numpy array (x1) and target variable to a numpy array (y).
   - Scale the features using StandardScaler().

4. Linear Regression Function:
   - Define linear_regression(X1, y) function for linear regression.
   - Add a column of ones to features for the intercept term.
   - Initialize theta as a zero vector.
   - Implement gradient descent to update theta.

5. Model Training and Prediction:
   - Call linear_regression function with scaled features (x1_scaled) and target variable (y).
   - Prepare new data for prediction by scaling and reshaping.
   - Use the optimized theta to predict the output for new data.

6. Print Prediction:
   - Inverse transform the scaled prediction to get the actual predicted value.
   - Print the predicted value.


## Program:
```py
#Program to implement the linear regression using gradient descent.
#Developed by: DINESH S
#RegisterNumber:  212222230033
```
### Linear Regression Function
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]

    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)

        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
```
### Read Data
```py
data=pd.read_csv("/content/50_Startups.csv")
data.head()
```
### Scaling
```py
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)
```
### Predicting
```py
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted valeue: {pre}")
```
## Output:
### Read Data
![image](https://github.com/SanjayRagavendar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/91368803/81f25156-61b0-4d4b-8431-3d8d455dfb49)

### Scaling
![image](https://github.com/SanjayRagavendar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/91368803/52edcd33-38cf-4dc5-a968-64d44b2a014b)

### Predicting
![image](https://github.com/SanjayRagavendar/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/91368803/3f81e9f8-c669-466f-bc3c-ce4f7c2e053b)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
