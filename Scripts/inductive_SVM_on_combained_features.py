#inductive SVM on the combained dataset (sequence features + atomic features)

import sklearn.svm as svm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_curve,f1_score,roc_auc_score

smv_clf = svm.SVC(C=1,kernel='linear')
smv_clf.fit(train.iloc[:,3:15],train.iloc[:,15])
#print("training error:",smv_clf.fit(x_train, np.ravel(y_train)).score(x_train,y_train))
svm_pred = smv_clf.predict(testset_0_1.iloc[:,3:])


f1 = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=svm_pred)
print("ACCURACY SVM " + str(f1))
print("F1 SVM  ",f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=svm_pred))
print("Precision=",precision_score(test_label_0_1.iloc[:,3],svm_pred))
print("Recall=",precision_score(test_label_0_1.iloc[:,3],svm_pred))
print("ROC score=", roc_auc_score(test_label_0_1.iloc[:,3],svm_pred))
print("                                ")



