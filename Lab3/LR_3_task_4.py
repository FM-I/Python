import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.5, random_state = 0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)


# Розрахунок коефіцієнтів регресії
coefficients = regr.coef_
intercept = regr.intercept_

# Розрахунок показників
r2 = r2_score(ytest, ypred)
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)

# Виведення коефіцієнтів регресії та показників на екран
print("Коефіцієнти регресії:")
print("Коефіцієнти: {}".format(coefficients))
print("Перетин: {}".format(intercept))
print("\nПоказники якості:")
print("R^2 Score: {}".format(r2))
print("Mean Absolute Error: {}".format(mae))
print("Mean Squared Error: {}".format(mse))


fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()