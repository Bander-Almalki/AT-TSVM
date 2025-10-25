print(testset_0_1)

# test set in in transfer. we have only sequence features

x_test_transfer_seq_only=np.copy(testset_0_1.iloc[:,3:13])

#Extracting the mean of atomic features of the training set
mask_at1_p=train.iloc[:,15]==1
mask_at1_n=train.iloc[:,15]==0

t_atomic_pca0_positive_mean=train[mask_at1_p].iloc[:,13].mean()
t_atomic_pca1_positive_mean=train[mask_at1_p].iloc[:,14].mean()

t_atomic_pca0_neg_mean=train[mask_at1_n].iloc[:,13].mean()
t_atomic_pca1_neg_mean=train[mask_at1_n].iloc[:,14].mean()

print(t_atomic_pca0_positive_mean)
print(t_atomic_pca1_positive_mean)
print(t_atomic_pca0_neg_mean)
print(t_atomic_pca1_neg_mean)


#Use inductive seq only labels as initial labels to assign the +ve,-ve train mean atomic features to the test set
print(svm3_pred)
predicted_tr_seq_label=pd.DataFrame(svm3_pred)
predicted_tr_seq_label=predicted_tr_seq_label.rename(columns={0:'label'})
print(predicted_tr_seq_label)

predicted_tr_seq_label["atomic0"]=0
predicted_tr_seq_label['atomic1']=0

#assigning the mean atomic features  based on the predicted seq label
predicted_tr_seq_label.loc[predicted_tr_seq_label["label"]==0, "atomic0"] = t_atomic_pca0_neg_mean
predicted_tr_seq_label.loc[predicted_tr_seq_label["label"]==1, "atomic0"] = t_atomic_pca0_positive_mean

predicted_tr_seq_label.loc[predicted_tr_seq_label["label"]==0, "atomic1"] = t_atomic_pca1_neg_mean
predicted_tr_seq_label.loc[predicted_tr_seq_label["label"]==1, "atomic1"] = t_atomic_pca1_positive_mean
mean_atomic_features=predicted_tr_seq_label.drop('label',axis=1)
mean_atomic_features=mean_atomic_features.to_numpy()

#concatinating seq_test set with the mean_atomic_train set
transfer_test_set=np.concatenate((x_test_transfer_seq_only,mean_atomic_features),axis=1)

indices = np.where(transfer_test_set[:,0])
indices
transfer_test_set2=np.copy(transfer_test_set)

