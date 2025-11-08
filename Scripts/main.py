import pandas as pd
import os
import Importing_data
import Choosing_random_train_test_examples
import inductive_SVM_on_combained_features
import inductive_SVM_on_seq_features
import inductive_SVM_on_atomic_features
import preparing_data_for_transfer
import Validation_set
import TSVM_with_KNN_Active_Learning_Call as TSVM_with_KNN_Active_Learning_Call
import Plotting

print("Importing Data ............")
train_data,train_label,test_data,test_label=Importing_data.read_data()

print("\n\nChoosing Random Train and Test examples ............\n")
train,testset,test_label,testset_0_1,test_label_0_1=Choosing_random_train_test_examples.random_data(train_data,train_label,test_data,test_label)

print("\n\nApplying inductive SVM on combained data ............\n")
svm_pred_comb= inductive_SVM_on_combained_features.inductive_SVM_comb(train,testset_0_1,test_label_0_1)

print("\n\nApplying inductive SVM on seq data ............\n")
svm3_pred=inductive_SVM_on_seq_features.inductive_svm_on_seq(train,testset_0_1,test_label_0_1)

print("\n\nApplying inductive SVM on atomic data ............\n")
s4vm_pred=inductive_SVM_on_atomic_features.inductive_svm_atomic(train,testset_0_1,test_label_0_1)

#preparing data for transfer
print("\n\nPreparing data for transfer ............\n")
transfer_test_set,transfer_test_set2,t_atomic_pca0_positive_mean,t_atomic_pca0_neg_mean,t_atomic_pca1_positive_mean,t_atomic_pca1_neg_mean=preparing_data_for_transfer.preparing_data_for_transfer(train,testset_0_1,test_label_0_1,svm3_pred)

print(transfer_test_set.shape)

#validation set 
print("\n\nApplying validation set ............\n")
random_rows10,random_rows20,random_rows30,random_rows40,random_rows50,random_rows75,sample10,sample20,sample30,sample40,sample50,sample75,partial_labels10,partial_labels20,partial_labels30,partial_labels40,partial_labels50,partial_labels75=Validation_set.validation(transfer_test_set,test_label,test_label_0_1)

print(sample10.shape)

#TSVM with Knn
print("\n Applying TSVM with knn:\n")
f1_iter,all_margin=TSVM_with_KNN_Active_Learning_Call.model_call(train,test_label_0_1,transfer_test_set,partial_labels10,partial_labels20,partial_labels30,partial_labels40,partial_labels50,partial_labels75,random_rows10,random_rows20,random_rows30,random_rows40,random_rows50,random_rows75,t_atomic_pca0_positive_mean,t_atomic_pca0_neg_mean,t_atomic_pca1_positive_mean,t_atomic_pca1_neg_mean)

#Ploatting
print("\n Plotting:\n")
Plotting.ploat(f1_iter,all_margin)
