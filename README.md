## Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
 
## Program:
```PY
#Program to implement the linear regression using gradient descent.
#Developed by: DINESH.S
#RegisterNumber: 212222230033

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))

```


## Output:
![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/d08dee80-df88-48d1-a354-6e0efde41bf2)

![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/f95cb657-dfc5-46ff-888b-751ad3d08aa7)

![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/cc1cfb50-558d-49a3-bd7c-c10b79140131)

![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/12056ea0-fb5b-4a30-935e-eb1f0ab57f5e)

![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/f6934e62-63ee-4d98-b622-a423dfd74877)

![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/138282ba-611f-4d70-b82c-670579285cd0)

![image](https://github.com/Daniel-christal/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145742847/a7562775-add0-4cad-8a38-49c9ab03e416)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
