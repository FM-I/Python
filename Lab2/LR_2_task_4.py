from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot

# Завантаження даних з файлу
url = "income_data.txt"
dataset = read_csv(url, header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                                            'marital_status', 'occupation', 'relationship', 'race', 'sex',
                                            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                                            'income'])

# Кодування категоріальних ознак
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                    'native_country', 'income']
for col in categorical_cols:
    dataset[col] = dataset[col].astype('category').cat.codes

# Розділення даних на ознаки (X) та відповіді (Y)
X = dataset.drop('income', axis=1)
Y = dataset['income']

# Розділення на навчальний та контрольний набори
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# Створення моделей класифікації
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(kernel='poly', degree=8)))

# Оцінка моделей за допомогою крос-валідації
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Порівняння моделей за допомогою графіку
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Вибір найкращої моделі (наприклад, SVM)
best_model = LinearDiscriminantAnalysis()
best_model.fit(X_train, Y_train)

# Оцінка найкращої моделі на контрольному наборі
predictions = best_model.predict(X_validation)
print("Точність на контрольному наборі: {}".format(accuracy_score(Y_validation, predictions)))
print("Матриця непорозумінь:")
print(confusion_matrix(Y_validation, predictions))
print("Звіт про класифікацію:")
print(classification_report(Y_validation, predictions))
