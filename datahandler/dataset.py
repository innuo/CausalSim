import torch
from torch.utils.data.dataset import Dataset 
import pandas as pd
from sklearn import preprocessing
import numpy as np
import re
import copy

class DataSet(Dataset): 
    def __init__(self, data_frame_list, infer_categoricals=True, categorical_cols=[]):
        """categorical_cols: list of categorical column _names_ """

        self.raw_datasets = data_frame_list
        self.df = pd.concat(data_frame_list, axis=0, ignore_index=True, sort=True)
        self.variable_names = list(self.df.columns)
 
        if not infer_categoricals:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = self.__sniff_categorical()

        for d in self.raw_datasets:
            for v in d:
                if v in self.categorical_cols:
                    d[v] = d[v].astype('category')
                else:
                    d[v] = d[v].astype(np.float64)
        
        self.variable_dict = dict()
        for i, v in enumerate(self.variable_names):
            self.variable_dict[v] = dict()
            self.variable_dict[v]['id'] = i
            if v in self.categorical_cols:
                self.variable_dict[v]['type'] = "categorical"
                le = preprocessing.LabelEncoder()
                inds = self.df[v].isnull() == False
                le.fit(self.df[v][inds])
                self.variable_dict[v]['dim'] = np.count_nonzero(le.classes_)
                self.variable_dict[v]['transform'] = le.transform
                self.variable_dict[v]['inverse_transform'] = le.inverse_transform
            else:
                self.variable_dict[v]['type'] = "numeric"
                self.variable_dict[v]['dim'] = 1
                scaler = preprocessing.StandardScaler()
                scaler.fit(self.df[v].values.reshape([-1, 1]))
                self.variable_dict[v]['transform'] = scaler.transform
                self.variable_dict[v]['inverse_transform'] = scaler.inverse_transform

        for v in self.variable_names:
            if(self.variable_dict[v]['type'] == 'categorical'):
                non_missing_inds = self.df[v].isnull() == False
                self.df[v][non_missing_inds] = self.variable_dict[v]['transform'](self.df[v][non_missing_inds])
                #self.df[v] = self.df[v].astype('category')
            else:
                self.df[v] = self.variable_dict[v]['transform'](self.df[v].values.reshape([-1, 1]))

    def __sniff_categorical(self):
        num_unique_values = [len(self.df[c].unique()) for c in list(self.df)]
        categorical_cols = [list(self.df)[i]
                            for i in range(len(num_unique_values))
                            if num_unique_values[i] < max(np.log(self.df.shape[0]), 10)]
        return categorical_cols

    def __getitem__(self, index):
        #slice_df = self.df.iloc[index].to_dict()
        slice_df = torch.tensor(self.df.iloc[index])
        return slice_df

    def __len__(self):
        return self.df.shape[0]

    pass


if __name__ == '__main__':
    #df = pd.read_csv("./data/5d.csv")
    dd = pd.read_csv('data/toy-DSD.csv', 
        usecols=['Product','Channel','Sales'])
    ds = pd.read_csv('data/toy-Survey.csv',
        usecols=['Product','Shopper','VolumeBought'])
    dp = pd.read_csv('data/toy-Product.csv',
        usecols = ['Product','Channel','Shopper'])

    dataset = DataSet([dd, ds, dp])
    print(dataset.df.describe())
    print(dataset.df.dtypes)