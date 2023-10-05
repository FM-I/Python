from sklearn import preprocessing
import numpy as np

Input_labels = ['red', 'Ыасk', 'red', 'green', 'Ьlack', 'yellow', 'white']

# Створення кодувальника та встановлення відповідності
# між мітками та числами
encoder = preprocessing.LabelEncoder()
encoder.fit(Input_labels)

print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_) :
    print(item, '-->', i)

test_labels = ['green', 'red', 'Ыасk']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list (encoded_values))

encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list (decoded_list ) )