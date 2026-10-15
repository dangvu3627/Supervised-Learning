import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pipeline import data_prep

X_train, y_train, _, _, X_test, y_test = data_prep()

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model = knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred=y_pred))
'''
                precision    recall  f1-score   support

         0.0       0.98      0.91      0.95       242
         1.0       0.79      0.94      0.86        77
         2.0       0.88      1.00      0.93        14
         3.0       0.86      0.92      0.89        13

    accuracy                           0.92       346
   macro avg       0.88      0.94      0.91       346
weighted avg       0.93      0.92      0.92       346
'''

