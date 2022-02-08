# Dataset used is from Basketball-Reference: https://www.basketball-reference.com/leagues/NBA_2021_per_game.html

import matplotlib.pyplot as plt
import numpy as np
import csv
import random

# Get data from csv file, and place it into lists
mp = []         # Feature (minutes played)
mp_squared = [] # Feature (minutes played squared)
pts = []        # Target variable (points scored)

with open('NBA_playerstats.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) # Skip the header
    for row in reader:
        mp.append(float(row[7]))
        mp_squared.append(float(row[7]) * float(row[7]))
        pts.append(float(row[29]))

# Plot the entire dataset:
plt.plot(mp, pts, 'b.')
plt.title('NBA Points Scored Per Minutes Played')
plt.xlabel('Minutes Played (Per Game)')
plt.ylabel('Points Scored (Per Game)')
plt.savefig('EntireDataSet.jpg')
plt.show()

# Generate training and validation sets.
def generate_sets(x):
    indexes = [i for i in range(len(x))]
    training_size = (len(x) // 5) * 4
    training = []
    for i in range(training_size):
        choice = random.choice(indexes)
        training.append(choice)
        indexes.remove(choice)

    # indexes is validation set
    return (training, indexes)

# Fit a linear regression model using batch gradient descent. 
# alpha = learning rate (scalar)
# x = features (matrix)
# y = target variable (matrix)
# stop = stopping point
def fit(alpha, x, y, stop):

    # n is the number of features, m is the number of training examples
    n, m = x.shape
    x_cat = np.concatenate((np.matrix(np.ones(m)), x))
    n, m = x_cat.shape
    theta = np.transpose(np.matrix(np.zeros(n)))

    while True:
        theta_old = np.copy(theta)
        h_x = np.dot(np.transpose(theta), x_cat)
        J = 0
        for i in range(n):
            for j in range(m):
                J += (h_x[0, j] - y[0, j]) * x_cat[i, j]
            theta[i] -= (alpha * J)
            J = 0

        if np.linalg.norm(theta - theta_old, ord=1) < stop:
            break

    return theta

def mean_squared_error(x, y, y_predicted):
    y = y.tolist()[0] 
    summation = 0
    for i in range(len(y)):
        summation += ((y[i] - y_predicted[i]) ** 2)
    return (1 / len(x)) * summation

sets = generate_sets(mp) # sets[0] is training indexes, sets[1] is validation indexes.

# LINEAR FIT

# Predict the amount of points a player will score (per game) based on how minutes played (per game).
x = [mp[sets[0][i]] for i in range(len(sets[0]))]
y = [pts[sets[0][i]] for i in range(len(sets[0]))]

x_linear = np.matrix([x])
y = np.matrix([y])
plt.plot(x_linear[0], y, 'b.')

# Decrease learning rate based on size of training set
theta_hat = fit(1e-8, x_linear, y, 1e-5) # Use this theta_hat for validation set.
x_linear = x_linear.tolist()[0]
y = [float(theta_hat[0, 0] + theta_hat[1, 0] * i) for i in x_linear]
plt.plot(x_linear, y, 'r.')
plt.title('Points per Minutes Played (Linear Fit)')
plt.xlabel('Minutes Played')
plt.ylabel('Points')
plt.savefig('LinearTraining.jpg')
plt.show()

# Calculate MSE using validation set
x = [mp[sets[1][i]] for i in range(len(sets[1]))]
y = [pts[sets[1][i]] for i in range(len(sets[1]))] 
x_linear = np.matrix([x])
y = np.matrix([y])
x_linear = x_linear.tolist()[0]
y_predicted = [float(theta_hat[0, 0] + theta_hat[1, 0] * i) for i in x_linear]
mse = '{0:.2f}'.format(mean_squared_error(x_linear, y, y_predicted))
y = y.tolist()[0]
plt.plot(x_linear, y, 'b.')
plt.plot(x_linear, y_predicted, 'r.')
plt.title('Points per Minutes Played (Linear Fit)')
plt.text(5, 20, 'Mean Squared Error = ' + str(mse), fontsize=16)
plt.xlabel('Minutes Played')
plt.ylabel('Points')
plt.savefig('LinearTest.jpg')
plt.show()


# QUADRATIC FIT
x = [mp[sets[0][i]] for i in range(len(sets[0]))]
x_squared = [mp[sets[0][i]] ** 2 for i in range(len(sets[0]))]
y = [pts[sets[0][i]] for i in range(len(sets[0]))]

# Predict the amount of points a player will score (per game) based on how minutes played (per game).
x_quadratic = np.matrix([x, x_squared])
y = np.matrix([y])
plt.plot(x_quadratic[0], y, 'b.')

# Decrease learning rate based on size of training set
theta_hat = fit(1e-8, x_quadratic, y, 1e-5) # Use this theta_hat for validation set.
y = [float(theta_hat[0, 0] + theta_hat[1, 0] * i + theta_hat[2, 0] * i * i) for i in x_quadratic.tolist()[0]]

plt.plot(x_quadratic.tolist()[0], y, 'r.')
plt.title('Points per Minutes Played (Quadratic Fit)')
plt.xlabel('Minutes Played')
plt.ylabel('Points')
plt.savefig('QuadraticTrain.jpg')
plt.show()

# Calculate MSE using validation set
x = [mp[sets[1][i]] for i in range(len(sets[1]))]
x_squared = [mp[sets[1][i]] ** 2 for i in range(len(sets[1]))]
y = [pts[sets[1][i]] for i in range(len(sets[1]))] 
x_quadratic = np.matrix([x, x_squared])
y = np.matrix([y])
y_predicted = [float(theta_hat[0, 0] + theta_hat[1, 0] * i + theta_hat[2, 0] * i * i) for i in x_quadratic.tolist()[0]]
mse = '{0:.2f}'.format(mean_squared_error(x_quadratic.tolist()[0], y, y_predicted))
y = y.tolist()[0]
plt.plot(x_quadratic.tolist()[0], y, 'b.')
plt.plot(x_quadratic.tolist()[0], y_predicted, 'r.')
plt.title('Points per Minutes Played (Quadratic Fit)')
plt.text(5, 20, 'Mean Squared Error = ' + str(mse), fontsize=16)
plt.xlabel('Minutes Played')
plt.ylabel('Points')
plt.savefig('QuadraticTest.jpg')
plt.show()
