#gammas=[0.001,0.01,0.2,0.4,0.8,1.5,3]
#cls=[1,2,5,10,20]
from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score

c=1
g=0.0001
y_test[y_test == 0] = -1
partial_labels10[partial_labels10 ==0]=-1
partial_labels20[partial_labels20 ==0]=-1
partial_labels30[partial_labels30 ==0]=-1
partial_labels40[partial_labels40 ==0]=-1
partial_labels50[partial_labels50 ==0]=-1
partial_labels75[partial_labels75 ==0]=-1



#print(clf1.coef_[0])
clf=TransductiveSVM(kernel="linear",Cl=c,Cu=0.5,X2=transfer_test_set2)
print("training error:",clf.fit(x_train, np.ravel(y_train),Y_True=y_test,max_queries=100,query_per_iter=5,
                                positive_mean_at0=t_atomic_pca0_positive_mean,
                                neg_mean_at0=t_atomic_pca0_neg_mean,positive_mean_at1=t_atomic_pca1_positive_mean,neg_mean_at1=t_atomic_pca1_neg_mean).score(x_train,y_train))

#clf.fit(X_train, np.ravel(y_train),Y_True= y_test2000)

y_predicted_active=clf.predict(transfer_test_set2)


f = accuracy_score(y_true=y_test, y_pred=y_predicted_active)
print("ACCURACY TSVM " + str(f))
print("F1 TSVM " + str(f1_score(y_true=y_test,y_pred=y_predicted_active)))
print("percision:",precision_score(y_true=y_test,y_pred=y_predicted_active))
print("recall:",recall_score(y_true=y_test,y_pred=y_predicted_active))
print("Roc:",roc_auc_score(y_test,y_predicted_active))
#print(clf.clf.coef_[0])

#print(y_svm)