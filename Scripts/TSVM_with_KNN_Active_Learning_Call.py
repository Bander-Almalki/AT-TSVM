
import numpy as np
import sklearn.svm as svm
import time
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
import math
import pandas as pd
import TSVM_with_KNN
import TSVM_with_KNN_Active_Learning as TSVM_with_KNN_Active_Learning


def model_call(train,test_label_0_1,transfer_test_set,partial_labels10,partial_labels20,partial_labels30,partial_labels40,partial_labels50,partial_labels75,random_rows10,random_rows20,random_rows30,random_rows40,random_rows50,random_rows75,t_atomic_pca0_positive_mean,t_atomic_pca0_neg_mean,t_atomic_pca1_positive_mean,t_atomic_pca1_neg_mean):

    
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
    clf=TSVM_with_KNN_Active_Learning.TransductiveSVM(kernel="linear",Cl=c,Cu=0.5,X2=transfer_test_set)
    # clf,f1_iter,all_margin,obj_fun=clf.fit(x_train, y_train.ravel(),Y_True=test_labels_00, partial_labels10=partial_labels10,partial_index10=random_rows10,
    #                                 partial_labels20=partial_labels20,partial_index20=random_rows20,
    #                                 partial_labels30=partial_labels30,partial_index30=random_rows30,
    #                                 partial_labels40=partial_labels40,partial_index40=random_rows40,
    #                                 partial_labels50=partial_labels50,partial_index50=random_rows50,
    #                                 partial_labels75=partial_labels75,partial_index75=random_rows75,
    #                                 positive_mean_at0=t_atomic_pca0_positive_mean,
    #                                 neg_mean_at0=t_atomic_pca0_neg_mean,positive_mean_at1=t_atomic_pca1_positive_mean,neg_mean_at1=t_atomic_pca1_neg_mean)


    clf,f1_iter,all_margin=clf.fit(x_train, np.ravel(y_train),Y_True=np.ravel(test_labels_00),max_queries=100,query_per_iter=5,
                                positive_mean_at0=t_atomic_pca0_positive_mean,
                                neg_mean_at0=t_atomic_pca0_neg_mean,positive_mean_at1=t_atomic_pca1_positive_mean,neg_mean_at1=t_atomic_pca1_neg_mean)

    #clf.fit(X_train, np.ravel(y_train),Y_True= y_test2000)

    y_predicted=clf.predict(transfer_test_set)


    f = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=y_predicted)
    print("ACCURACY TSVM " + str(f))
    print("F1 TSVM " + str(f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted)))
    print("precision:",precision_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted))
    print("recall:",recall_score(y_true=test_label_0_1.iloc[:,3],y_pred=y_predicted))
    print("ROC:",roc_auc_score(test_label_0_1.iloc[:,3],y_predicted))

 
    #return
    return f1_iter,all_margin


