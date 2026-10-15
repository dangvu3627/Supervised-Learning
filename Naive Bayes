from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from pipeline import data_prep

X_train, y_train, _, _, X_test, y_test = data_prep()

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print(classification_report(y_test, y_pred=y_pred))

"""
- High recall for 3.0
                precision    recall  f1-score   support

         0.0       0.98      0.77      0.86       242
         1.0       0.60      0.48      0.53        77
         2.0       0.59      0.71      0.65        14
         3.0       0.17      1.00      0.29        13

    accuracy                           0.71       346
   macro avg       0.58      0.74      0.58       346
weighted avg       0.85      0.71      0.76       346
"""
