c=1
g=0.0001

mask = (test_label['sequence_index'] == seq_number) & (test_label['label'] == 0)
test_label.loc[mask, 'label'] = -1
#test_label_grouped.get_group(seq_number).iloc[:,3][test_label_grouped.get_group(seq_number).iloc[:,3] == 0] = -1


seq_only_tsvm=TransductiveSVM_seq_only(kernel="linear",Cl=c,Cu=0.5,X2=testset_0_1.iloc[:,3:13])
print("training error:",seq_only_tsvm.fit(train.iloc[:,3:13], np.ravel(train.iloc[:,15]),Y_True=test_label_0_1.iloc[:,3]).score(train.iloc[:,3:13],train.iloc[:,15]))

#clf.fit(X_train, np.ravel(y_train),Y_True= y_test2000)

y_predicted_seq=seq_only_tsvm.predict(testset_0_1.iloc[:,3:13])


f = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=y_predicted_seq)
print("ACCURACY TSVM " + str(f))
print("F1 TSVM " + str(f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted_seq)))
print("precision=",precision_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted_seq))
print("recall=",recall_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted_seq))
print("ROC=",roc_auc_score(test_label_0_1.iloc[:,3],y_predicted_seq))




