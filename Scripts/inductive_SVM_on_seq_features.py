import sklearn.svm as svm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_curve,f1_score,roc_auc_score
import top_l_precisions

def inductive_svm_on_seq(train,testset_0_1,test_label_0_1):

    # Inductive SVM on seq features only
    smv_clf3 = svm.SVC(C=1,kernel='linear',probability=True)
    smv_clf3.fit(train.iloc[:,3:13], np.ravel(train.iloc[:,15]))
    print("training error:",smv_clf3.fit(train.iloc[:,3:13], np.ravel(train.iloc[:,15])).score(train.iloc[:,3:13],np.ravel(train.iloc[:,15])))
    svm3_pred = smv_clf3.predict(testset_0_1.iloc[:,3:13])
    svm3_pred_proba=smv_clf3.predict_proba(testset_0_1.iloc[:,3:13])
    svm_pred_proba=svm3_pred_proba[:,1]

    f3 = accuracy_score(y_true=test_label_0_1.iloc[:,3], y_pred=svm3_pred)
    print("ACCURACY SVM " + str(f3))
    print("F1 SVM  ",f1_score(y_true=test_label_0_1.iloc[:,3],y_pred=svm3_pred))
    print("Precision=",precision_score(test_label_0_1.iloc[:,3],svm3_pred))
    print("Recall=",recall_score(test_label_0_1.iloc[:,3],svm3_pred))
    print("ROC score=", roc_auc_score(test_label_0_1.iloc[:,3],svm3_pred))
    print("\n")

    y_t=test_label_0_1.iloc[:,3]
    y_t.reset_index(drop=True, inplace=True)

    precisions=top_l_precisions.calculate_top_l_precisions(svm_pred_proba,y_t)
    print("top_l precision=",precisions['Top-L'])
    print("top_l/2 precision=",precisions['Top-L/2'])
    print("top_l/5 precision=",precisions['Top-L/5'])
    print("top_l/10 precision=",precisions['Top-L/10'])
    
    return svm3_pred



