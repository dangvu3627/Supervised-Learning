from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pipeline import data_prep

X_train, y_train, _, _, X_test, y_test = data_prep()

lg_model = LogisticRegression() # Could be tuned
lg_model = lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)

print(classification_report(y_test, y_pred=y_pred))
'''
              precision    recall  f1-score   support

         0.0       0.96      0.83      0.89       242
         1.0       0.69      0.79      0.73        77
         2.0       0.61      1.00      0.76        14
         3.0       0.48      0.92      0.63        13

    accuracy                           0.83       346
   macro avg       0.68      0.89      0.75       346
weighted avg       0.87      0.83      0.84       346
'''