import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Згенерувати випадкові дані
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)

# Побудувати модель лінійної регресії
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Побудувати модель поліноміальної регресії
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Оцінити якість поліноміальної регресії
y_pred_poly = poly_reg.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

# Вивести значення коефіцієнтів полінома
coefficients_poly = poly_reg.coef_[0]
intercept_poly = poly_reg.intercept_

print("Коефіцієнти полінома: {}".format(coefficients_poly))
print("Перетин полінома: {}".format(intercept_poly))

# Вивести графіки
plt.figure(figsize=(10, 6))

# Графік випадкових даних
plt.scatter(X, y, color='blue', label='Випадкові дані')

# Графік поліноміальної регресії
X_range = np.arange(-3, 3, 0.1).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
y_pred_range_poly = poly_reg.predict(X_range_poly)
plt.plot(X_range, y_pred_range_poly, color='red', label='Поліноміальна регресія')

# Підписи та відображення графіку
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Поліноміальна регресія та випадкові дані')
plt.show()

# Вивести якість поліноміальної регресії
print("Середньоквадратична похибка поліноміальної регресії: {}".format(mse_poly))
