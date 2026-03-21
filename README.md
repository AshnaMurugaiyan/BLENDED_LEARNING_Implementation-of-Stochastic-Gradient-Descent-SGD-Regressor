# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets.
6. Create an SGD Regressor model.
7. Fit the model on the training data.
8. Evaluate the model performance.
9. Make predictions and visualize the results.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Ashna M
RegisterNumber: 212225040032
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
x=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_model = SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(x_train,y_train)
y_pred=sgd_model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print("="*50)
print('Name: Ashna M')
print('Reg No:212225040032')
print(f"MSE: {mse:.4f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print("="*50)
print("Model Coefficients:")
print("Coefficiens:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
*/
```

## Output:
![alt text](<Screenshot 2026-03-21 151739.png>)
<img width="1002" height="682" alt="Screenshot 2026-03-21 150731" src="https://github.com/user-attachments/assets/6f2c5dc7-73c7-438f-b7ad-e7ee9298cf88" />
<img width="1046" height="433" alt="Screenshot 2026-03-21 150745" src="https://github.com/user-attachments/assets/ceb789f3-27d4-48f9-9884-1a4b2d138d19" />
<img width="1074" height="569" alt="Screenshot 2026-03-21 150808" src="https://github.com/user-attachments/assets/4c355271-ff51-41c2-9343-c4312aa75b38" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
