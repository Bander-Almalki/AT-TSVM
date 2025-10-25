precentage10=int(transfer_test_set.shape[0]*0.10)
precentage20=int(transfer_test_set.shape[0]*0.20)
precentage30=int(transfer_test_set.shape[0]*0.30)
precentage40=int(transfer_test_set.shape[0]*0.40)
precentage50=int(transfer_test_set.shape[0]*0.50)
precentage75=int(transfer_test_set.shape[0]*0.75)
precentage50

random_rows10 = np.random.choice(transfer_test_set.shape[0], size=precentage10, replace=False)
random_rows20 = np.random.choice(transfer_test_set.shape[0], size=precentage20, replace=False)
random_rows30 = np.random.choice(transfer_test_set.shape[0], size=precentage30, replace=False)
random_rows40 = np.random.choice(transfer_test_set.shape[0], size=precentage40, replace=False)
random_rows50 = np.random.choice(transfer_test_set.shape[0], size=precentage50, replace=False)
random_rows75 = np.random.choice(transfer_test_set.shape[0], size=precentage75, replace=False)


sample10 = transfer_test_set[random_rows10, :]
sample20 = transfer_test_set[random_rows20, :]
sample30 = transfer_test_set[random_rows30, :]
sample40 = transfer_test_set[random_rows40, :]
sample50 = transfer_test_set[random_rows50, :]
sample75 = transfer_test_set[random_rows75, :]

sample10.shape

#group test lable
test_label_grouped=test_label.groupby('sequence_index')

test_lable_0=pd.concat([test_label_grouped.get_group(18)])
test_lable_0

partial_labels10=test_label_0_1['label'].iloc[random_rows10]
partial_labels20=test_label_0_1['label'].iloc[random_rows20]
partial_labels30=test_label_0_1['label'].iloc[random_rows30]
partial_labels40=test_label_0_1['label'].iloc[random_rows40]
partial_labels50=test_label_0_1['label'].iloc[random_rows50]
partial_labels75=test_label_0_1['label'].iloc[random_rows75]

partial_labels10

