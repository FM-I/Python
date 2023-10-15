import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

iris = load_iris()
X, y = iris.data, iris.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

# Класифікація з використанням класифікатора Ridge
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

# Оцінка якості класифікації
accuracy = metrics.accuracy_score(ytest, ypred)
precision = metrics.precision_score(ytest, ypred, average='weighted')
recall = metrics.recall_score(ytest, ypred, average='weighted')
f1_score = metrics.f1_score(ytest, ypred, average='weighted')
cohen_kappa = metrics.cohen_kappa_score(ytest, ypred)
matthews_corrcoef = metrics.matthews_corrcoef(ytest, ypred)

print('Accuracy:', np.round(accuracy, 4))
print('Precision:', np.round(precision, 4))
print('Recall:', np.round(recall, 4))
print('F1 Score:', np.round(f1_score, 4))
print('Cohen Kappa Score:', np.round(cohen_kappa, 4))
print('Matthews Corrcoef:', np.round(matthews_corrcoef, 4))

# Візуалізація матриці непорозумінь
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig("Confusion.jpg")

# Збереження SVG в об'єкті BytesIO
f = BytesIO()
plt.savefig(f, format="svg")
