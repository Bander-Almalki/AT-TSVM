#Inductive SVM on atomic features only
smv_clf4 = svm.SVC(C=1,kernel='linear',probability=True)
smv_clf4.fit(train.iloc[:,13:15], np.ravel(train.iloc[:,15]))
print("training error:",smv_clf4.fit(train.iloc[:,13:15], np.ravel(train.iloc[:,15])).score(train.iloc[:,13:15],train.iloc[:,15]))
svm4_pred = smv_clf4.predict(testset_0_1.iloc[:,13:15])


f3 = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=svm4_pred)
print("ACCURACY SVM " + str(f3))
print("F1 SVM  ",f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=svm4_pred))
print("                                ")
