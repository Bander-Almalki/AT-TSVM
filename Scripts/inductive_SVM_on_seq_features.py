# Inductive SVM on seq features only
smv_clf3 = svm.SVC(C=1,kernel='linear',probability=True)
smv_clf3.fit(train.iloc[:,3:13], np.ravel(train.iloc[:,15]))
print("training error:",smv_clf3.fit(train.iloc[:,3:13], np.ravel(train.iloc[:,15])).score(train.iloc[:,3:13],np.ravel(train.iloc[:,15])))
svm3_pred = smv_clf3.predict(testset_0_1.iloc[:,3:13])
svm3_pred_proba=smv_clf3.predict_proba(testset_0_1.iloc[:,3:13])

f3 = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=svm3_pred)
print("ACCURACY SVM " + str(f3))
print("F1 SVM  ",f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=svm3_pred))
print("Precision=",precision_score(test_label_0_1.iloc[:,3],svm3_pred))
print("Recall=",recall_score(test_label_0_1.iloc[:,3],svm3_pred))
print("ROC score=", roc_auc_score(test_label_0_1.iloc[:,3],svm3_pred))

calculate_top_l_precisions(svm_pred_proba,y_t)
