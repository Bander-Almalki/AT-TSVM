
import pandas as pd


def random_data(train_data,train_label,test_data,test_label):   
    
     # Choosing Train Examples

    #concat train_data and label
    trainset=pd.concat([train_data,train_label.iloc[:,3]],axis=1)
    trainset.head()

    #contact and noncontact examples
    contact=trainset[trainset['label']==1]
    contact.shape

    no_contact=trainset[trainset['label']==0]
    no_contact.shape

    train=pd.concat([contact.sample(n=500),no_contact.sample(n=500)])


    # Choosing Test Examples
    testset=pd.concat([contact[~contact.index.isin(train.index)].sample(n=2000),no_contact[~no_contact.index.isin(train.index)].sample(n=3000)])
    print(testset.shape)
    print(testset.columns)

    #group test_data by sequence_index
#     testset=test_data.groupby('sequence_index')
#     test_label_grouped=test_label.groupby('sequence_index')

    # prompt: choose rows from testset where sequence_index =

#     testset_0_1=pd.concat([testset.get_group(18)])
#     test_label_0_1=pd.concat([test_label_grouped.get_group(18)])
    testset_0_1=testset.drop(['label'],axis=1)
    test_label_0_1=testset[['sequence_index', 'position_1', 'position_2','label']]
    print(testset_0_1.columns)
    print(test_label_0_1.columns)

    return train,testset,test_label,testset_0_1,test_label_0_1