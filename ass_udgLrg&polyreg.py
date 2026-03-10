# Salary Prediction

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# Get the Data
year_of_experience = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7,
                      3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7,
                      6.0, 6.3, 6.6, 6.9, 7.2, 7.5, 7.8, 8.1, 8.4, 8.7,
                      9.0, 9.3, 9.6, 9.9, 10.2, 10.5, 10.8, 11.1, 11.4, 11.7,
                      12.0, 12.3, 12.6, 12.9, 13.2, 13.5, 13.8, 14.1, 14.4, 14.7]
salary = [30200, 31100, 32500, 33000, 34800, 36000, 37200, 38100, 39000, 40500,
          41800, 43000, 44200, 45000, 46000, 47100, 48000, 48800, 49500, 50000,
          50300, 50700, 51000, 51200, 51500, 51600, 51700, 51800, 51900, 52000,
          52100, 52000, 51900, 51800, 51700, 51500, 51200, 50900, 50500, 50000,
          49400, 48700, 48000, 47200, 46300, 45500, 44600, 43700, 42800, 41900]

# converting the data into np array to reshape it

X = np.array(year_of_experience).reshape(-1, 1)
y = np.array(salary)

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# creating the model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
 
# Predictions
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# Errors
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Linear Regression:")
print(f"Training MSE:{train_mse}")
print(f"Testing MSE:{test_mse}")

# Polynomial regression with increasing degrees

degrees = [1, 2, 3, 5, 8, 10, 12, 14, 16]

train_errors = []
test_errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    print(f"Degree {d} -> Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")

# Plot1: Model Fits vs Data

X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Data", alpha=0.6)

for d in [1, 3, 10]:
    poly = PolynomialFeatures(degree=d)
    
    # Fit only on training data
    X_train_poly = poly.fit_transform(X_train)
    X_plot_poly = poly.transform(X_plot)   # transform, NOT fit_transform
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, label=f"Degree {d}")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression Fits for Different Degrees")
plt.legend()
plt.show()


# Plot 2: Training Error vs Testing Error

plt.figure(figsize=(8, 5))

plt.plot(degrees, train_errors, marker='*', label="Training Error")
plt.plot(degrees, test_errors, marker='*', label="Testing Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error (Bias–Variance Tradeoff)")
plt.legend()
plt.show()

print("\n Model has successfully excuted")


# The questions mentioned in pdf : 

# 1. Why does training error always decrease with higher polynomial degree?

# As polynomial degree increases, the model gains more parameters and flexibility, and a higher-degree 
# polynomial can represent all lower-degree models. This allows the model to fit the training data more closely, 
# which leads to a monotonic decrease in training error.

# 2. Why does test error behave differently?

# In this experiment, the testing error decreases initially as the polynomial degree increases, because 
# the model reduces bias and better captures the non-linear relationship. However, beyond degree 10, the testing 
# error increases sharply, indicating that the model starts overfitting the training data and suffers from high 
# variance, leading to poor generalization.

# 3. At what point does the model start overfitting, and how can you tell?

# Overfitting starts at degree 12.

# 4. If you had to choose one polynomial degree, which would it be and why?

# Degree 10 should be chosen, because it achieves the minimum testing error
# and represents the best balance between bias and variance
