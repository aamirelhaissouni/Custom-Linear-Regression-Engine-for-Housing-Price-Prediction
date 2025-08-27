import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from debugpy.common.timestamp import current

# Load datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


## Specify features and target column of Prices
feature = ['GrLivArea', 'FullBath', 'TotalBsmtSF', 'BedroomAbvGr']
goal = 'Price'

## Data Preperation
xTrain = train_data.drop(columns=['Id','Price']).values
yTrain = train_data['Price'].values

xTest = test_data.drop(columns=['Id','Price']).values
yTest = test_data['Price'].values

# Prediction function
def pred(X, W):
    ## Predict prices with X/W
    return np.dot(X, W)

# Loss function (Mean Squared Error)
def loss(y_pred, y):
    return np.mean((y_pred - y) ** 2)

# Gradient calculation
def gradient(X, y_pred, y):
    m = y.shape[0]
    mse_error = y_pred-y
    return (2 / m) * np.dot(X.T,mse_error)


# Update function
def update(W, grad, alpha):
    return W - alpha * grad


# Training loop without debugging anymore
def train(X, y, alpha, n_iterations):
    W = np.zeros(X.shape[1]) # Initialize as zeros so gradient curve is shown properly (thank you TA Aiden Flood)
    losses = []
    for i in range(n_iterations):
        y_pred = pred(X, W)  # Predictions
        current_loss = loss(y_pred, y)  # Calculate loss
        losses.append(current_loss)

        grad = gradient(X, y_pred, y)  # Calculate gradient
        W = update(W, grad, alpha)  # Update weights
    print(f"MSE:  {current_loss}")
    print(f'Final weights:  {W}')
    return losses

## Alpha value (Question 10)
alpha1 = 0.2
n_iterations10 = 200000

## See what happens with alpha a = 0.2
train(xTrain,yTrain,alpha1, n_iterations10)
## Results in an overflow error because of large alpha value, rest of code should run like normal

## Part 2 Task 10:
## No the minimal MSE is not found, it converges at around 30000 iterations, because the alpha is so large the model
## overshoots, and an overflow error occurs.

# Part 2 Task 11: Train with α = 1e-11 and α = 1e-12, plot learning curves
alphas = [1e-11, 1e-12]
"""
n_iterations11 = 350000
a1 = train(xTrain,yTrain,alphas[0],n_iterations11)
a2 = train(xTrain,yTrain,alphas[1],n_iterations11)


# Plot learning curves for Task 11-12
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations11), a1, label=r'$\alpha = 10^{-11}$')
plt.plot(range(n_iterations11), a2, label=r'$\alpha = 10^{-12}$')
plt.title("Learning Curves for α = 1e-11 and α = 1e-12")
plt.xlabel("Iterations")
plt.ylabel("MSE/Loss")
plt.legend()
plt.grid(True)
plt.show()
"""
## Part 2 Task 13
## Use one of the same alphas as task 12
al = alphas[0]
## Now find the weights using the test_data and same iterations as tasks 11-12
Weights_test = train(xTest,yTest,al,350000)
## Then predict
print("Predicted house prices: " + pred(xTest, Weights_test))

## The MSE using the train data set will be slightly better as the model is more used to the train data set
## and may have an ever so slight bias to those numbers/data set



