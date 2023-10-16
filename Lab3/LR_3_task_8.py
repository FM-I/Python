from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import  KMeans

iris = load_iris()
X = iris['data']
y = iris['target']
n_clusters = 5
rseed = 0

#Створення об'єкту KMeans для кластеризації з вказаною кількістю кластерів та налаштуваннями
kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = "warn", max_iter = 300, tol = 0.0001,  verbose = 0, random_state = None, copy_x = True,  algorithm = 'lloyd')
#Навчання моделі KMeans на даних
kmeans.fit(X)
#Передбачення кластерів для кожного зразка вхідних даних
y_kmeans = kmeans.predict(X)

# Відображення даних та кластерних центрів на графіку
plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 200, alpha = 0.5);
plt.show()

# Функція для знаходження кластерів вручну
def find_clusters(X, n_clusters, rseed = 2):
    # Ініціалізація рандомних центрів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]

    centers = X[i]

    while True:

        # Призначення кожному зразку найближчого кластера
        labels = pairwise_distances_argmin(X, centers)
        # Обчислення нових центрів кластерів
        new_centers = np.array([X[labels == j].mean(0) for j in range(n_clusters)])
        # Перевірка, чи центри не змінились
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels


# Знаходження та відображення кластерів, знайдених вручну для 3 кластерів з іншим seed
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c = labels, s = 50, cmap = 'viridis')
plt.show()

# Знаходження та відображення кластерів, знайдених вручну для 3 кластерів з іншим seed
centers, labels = find_clusters(X, 3, rseed = 0)
plt.scatter(X[:, 0], X[:, 1], c = labels, s = 50, cmap = 'viridis')
plt.show()

# Кластеризація за допомогою KMeans з вбудованим методом fit_predict для 3 кластерів
labels = KMeans(3, random_state = 0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c = labels, s = 50, cmap = 'viridis')
plt.show()
