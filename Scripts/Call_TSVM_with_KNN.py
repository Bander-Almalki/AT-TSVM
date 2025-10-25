#gammas=[0.001,0.01,0.2,0.4,0.8,1.5,3]
#cls=[1,2,5,10,20]
c=1
g=0.0001
#test_label_grouped.get_group(seq_number).iloc[:,3][test_label_grouped.get_group(seq_number).iloc[:,3] == 0] = -1

# partial_labels10[partial_labels10 ==0]=-1
# partial_labels20[partial_labels20 ==0]=-1
# partial_labels30[partial_labels30 ==0]=-1
# partial_labels40[partial_labels40 ==0]=-1
# partial_labels50[partial_labels50 ==0]=-1
# partial_labels75[partial_labels75 ==0]=-1


x_train=train.iloc[:,3:15].to_numpy()
y_train=train.iloc[:,15].to_numpy()

test_labels_00=test_label_0_1.iloc[:,3]
test_labels_00[test_labels_00 ==0]=-1
partial_labels10[partial_labels10 ==0]=-1
partial_labels20[partial_labels20 ==0]=-1
partial_labels30[partial_labels30 ==0]=-1
partial_labels40[partial_labels40 ==0]=-1
partial_labels50[partial_labels50 ==0]=-1
partial_labels75[partial_labels75 ==0]=-1




#print(clf1.coef_[0])
clf=TransductiveSVM(kernel="linear",Cl=c,Cu=0.5,X2=transfer_test_set)
print("training error:",clf.fit(x_train, y_train.ravel(),Y_True=test_labels_00, partial_labels10=partial_labels10,partial_index10=random_rows10,
                                partial_labels20=partial_labels20,partial_index20=random_rows20,
                                partial_labels30=partial_labels30,partial_index30=random_rows30,
                                partial_labels40=partial_labels40,partial_index40=random_rows40,
                                partial_labels50=partial_labels50,partial_index50=random_rows50,
                                partial_labels75=partial_labels75,partial_index75=random_rows75,
                                positive_mean_at0=t_atomic_pca0_positive_mean,
                                neg_mean_at0=t_atomic_pca0_neg_mean,positive_mean_at1=t_atomic_pca1_positive_mean,neg_mean_at1=t_atomic_pca1_neg_mean).score(x_train,y_train))

#clf.fit(X_train, np.ravel(y_train),Y_True= y_test2000)

y_predicted=clf.predict(transfer_test_set)


f = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=y_predicted)
print("ACCURACY TSVM " + str(f))
print("F1 TSVM " + str(f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted)))
print("precision:",precision_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted))
print("recall:",recall_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted))
print("ROC:",roc_auc_score(test_label_0_1.iloc[:,3],y_predicted))

#print(clf.clf.coef_[0])

#print(y_svm)