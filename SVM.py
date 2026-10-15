from sklearn.svm import SVC
from sklearn.metrics import classification_report
from pipeline import data_prep

X_train, y_train, _, _, X_test, y_test = data_prep()

svm_model = SVC() # Could be tuned
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred=y_pred))
'''
HIGHEST RESULT IN ALL CLASSIFICATION METHOD USED SO FAR
             precision    recall  f1-score   support

         0.0       1.00      0.96      0.98       242
         1.0       0.88      0.97      0.93        77
         2.0       0.93      0.93      0.93        14
         3.0       0.87      1.00      0.93        13

    accuracy                           0.96       346
   macro avg       0.92      0.97      0.94       346
weighted avg       0.97      0.96      0.96       346
'''
