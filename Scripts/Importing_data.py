import pandas as pd
import os

def read_data():

    current_path=os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)      # One level up

    #import train_data
    train_data=pd.read_csv(parent_path+'/Data/train_seq_top10_plus_atomic_top2_by_average_precision.csv')
    train_label=pd.read_csv(parent_path+'/Data/seq_train_label.csv')
    train_data.drop(['Unnamed: 0'],axis=1,inplace=True)

    #import test_data
    test_data=pd.read_csv(parent_path+'/Data/test_seq_top10_plus_atomic_top2_by_average_precision.csv')
    test_label=pd.read_csv(parent_path+'/Data/seq_test_label.csv')
    test_data.drop(['Unnamed: 0'],axis=1,inplace=True)
    #group by sequence_index
    test_grouped=test_data.groupby('sequence_index')
    #size of each group
    print(test_grouped.size())

    return train_data,train_label,test_data,test_label  