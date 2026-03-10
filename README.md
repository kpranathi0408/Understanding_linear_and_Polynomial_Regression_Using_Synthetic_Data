Bias–Variance Tradeoff Analysis using Polynomial Regression
Assignment Overview

This assignment analyzes how model complexity affects training and testing error using Linear Regression and Polynomial Regression.   
The objective is to observe the bias–variance tradeoff by increasing the polynomial degree and comparing model performance.   

Source code: 
ass_udgLrg&polyreg

Dataset

The dataset contains two variables:
Years of Experience
Salary
The goal is to predict salary based on years of experience.

Example values:
Experience	Salary
0.0	30200
3.0	41800
7.5	51600
12.0	49400   

Methodology
1. Data Preparation

Convert the dataset into NumPy arrays
Split data into training and testing sets (80% training, 20% testing)

Example:
train_test_split(X, y, test_size=0.2)
2. Linear Regression
A linear regression model is trained as the baseline model.

Example:
model = LinearRegression()
model.fit(X_train, y_train)

Training and testing errors are calculated using Mean Squared Error (MSE).

3. Polynomial Regression

Polynomial regression models are trained using different degrees:
[1, 2, 3, 5, 8, 10, 12, 14, 16]
Higher degrees increase model flexibility and allow the model to capture non-linear patterns.

Example:

PolynomialFeatures(degree=d)
Evaluation Metric
Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values.

Lower MSE indicates better model performance.

Results

Observations from the experiment:
Training error decreases as polynomial degree increases because the model becomes more flexible.
Testing error initially decreases but increases after a certain degree, indicating overfitting.

Examples:
Underfitting
Degree 1
Model too simple
Overfitting
Degree 12+
Model memorizes training data
Overfitting Point
The model begins overfitting at degree 12, where testing error increases significantly.   

Best Model

The best performing model is:
Polynomial Degree = 10

Reason:
Minimum testing error  
Balanced bias and variance   

Output 
The program generates two plots:  
Regression curves for different polynomial degrees  
Training error vs testing error  
These plots illustrate the bias–variance tradeoff.
