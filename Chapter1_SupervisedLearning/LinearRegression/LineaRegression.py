import matplotlib
import pandas as pd
import numpy as np
from scipy import stats
from math import exp, sqrt, pi

import datetime
from datetime import datetime, date

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import seaborn as sns
from pylab import rcParams

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pathFile = "Resources/pizza.txt"
    X, Y = np.loadtxt(pathFile, skiprows=1, unpack=True)  # X contains the values of the input variable, and Y contains the labels.
    print(X[0:5])  # [13.  2. 14. 23. 13.]
    print(Y[0:5])  # [33. 16. 32. 51. 27.]

    # Plotting data
    sns.set()  # activate Seaborn
    plt.axis([0, 50, 0, 50])  # scale axes (0 to 50)
    plt.xticks(fontsize=15)  # set x axis ticks
    plt.yticks(fontsize=15)  # set y axis ticks
    plt.xlabel("Reservations", fontsize=30)  # set x axis label
    plt.ylabel("Pizzas", fontsize=30)  # set y axis label
    plt.plot(X, Y, "bo")  # plot data
    plt.show()  # display chart

    # Defining the Model
    # To turn linear regression into running code, we need some way to represent
    # the line numerically. That’s where mathematics comes into the picture.
    # Here is the mathematical equation of a line that passes by the origin of the axes:
    # y = x * w
    # You might remember this equation from your studies, but don’t fret if not.
    # Here is what it means: each line passing by the origin is uniquely identified
    # by a single value. I called this value w, short for weight. You can also think
    # of w as the SLOPE of the line: the larger w is, the steeper the line. Check out
    # the following graph:
    # Before we move on, let me rewrite the equation of the line with a slightly dif-
    # ferent notation:
    # ŷ = x * w
    # I changed the symbol y to ŷ (read “y-hat”), because I don’t want to confuse
    # this value with the y values that we loaded from Roberto’s file. Both symbols
    # represent the number of pizzas, but there is a crucial difference between the
    # two:
    # ŷ  - a forecast—our prediction of how many pizzas we hope to sell.
    # y  - are real-life observations— what machine learning practictioners call the 'ground truth'.

    def predict(X, w):
        # predicts the pizzas from the reservations.
        return X * w

    # Implementing Training
    # Loss Function - a function that takes the examples (X and Y) and a line (w), and measures the line’s error.
    w = 1.5
    error = predict(X, w) - Y
    SE = error ** 2  # squared_error
    print(SE)

    def loss(X, Y, w):
        return np.average((predict(X, w) - Y) ** 2)

    MSE = loss(X, Y, w)  # Mean Squared Error
    print(MSE)  # 94.96666666666667

    def train(X: np.ndarray,
              Y: np.ndarray,
              iterations: int,
              learning_rate: float
              ) -> float:
        # We just added the learning rate to w, that results in a new line. Does this new
        # line result in a lower loss than our current line? If so, then w + lr becomes the
        # new current w, and the loop continues. Otherwise, the algorithm tries another
        # report erratum • discussCoding Linear Regression • 25
        # line: w - lr. Once again, if that line results in a lower loss than the current w,
        # the code updates w and continues the loop.
        # If neither w + lr nor w - lr yield a better loss than the current w, then we’re done.
        # We’ve approximated the examples as well as we can, and we return w to the
        # caller.

        w = 0
        for i in range(iterations):
            current_loss = loss(X, Y, w)
            print("Iteration %4d => Loss: %.6f" % (i, current_loss))

            if (loss(X, Y, w + learning_rate) < current_loss):
                w += learning_rate
            elif (loss(X, Y, w - learning_rate) < current_loss):
                w -= learning_rate
            else:
                return w


    # Train the system
    w = train(X, Y, iterations=10_000, learning_rate=0.01)
    print("\nw=%.3f" % w)

    # Predict the number of pizzas
    print("Prediction for %d reservations is %.2f pizzas" % (20, predict(20, w)))
    # w=1.844
    # Prediction for 20 reservations is 36.88 pizzas









    # Coding Linear Regression


