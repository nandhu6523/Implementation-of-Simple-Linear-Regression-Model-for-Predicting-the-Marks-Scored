# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
 1.import the needed packages.

 2. Assigning hours to x and scores to y.

 3. Plot the scatter plot.

 4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:S Nandhini 
RegisterNumber: 212222220028 
*/
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
Head:

   ![Screenshot 2023-10-19 102009](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/c42dfcd6-f577-48be-af33-3c84f488acc0)

Tail:

  ![Screenshot 2023-10-19 102018](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/1c773026-6ea4-4b7e-b42a-adf9c6844f8e)

Array value of x:

  ![Screenshot 2023-10-19 102037](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/756b18c7-c3b6-43c4-b587-7d34bff8deb6)

Array value of y:

  ![Screenshot 2023-10-19 102046](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/8a106a11-5038-4c9c-b3e9-a2e106da61d0)

Y prediction:

 ![Screenshot 2023-10-19 102451](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/d1b35d36-2b29-4dbf-bb45-0f79331353f3)

Array value of y test:

 ![Screenshot 2023-10-19 102546](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/f4a351a9-341b-4f75-bcca-e2adf90408f4)

Training set graph:

![Screenshot 2023-10-19 102119](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/2996c832-5531-4440-971c-ad520613ad3c)

Testing set graph:

![Screenshot 2023-10-19 102130](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/17ad5f72-0969-4b28-b30d-6ea0591b8199)

MSE MAE AND RMSE:

![Screenshot 2023-10-19 102138](https://github.com/nandhu6523/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123856724/f0f3d787-03e5-459e-85e7-80f1cc28a550)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
