from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from sklearn import preprocessing
import numpy as np
import re
import copy

class MyMixedDataSet(Dataset):

    def __init__(self, raw_dataset, input_cols=None, output_cols=None):
        self.raw_dataset = raw_dataset
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        output_vec = self.raw_dataset.df.loc[index, self.output_cols]
        input_vec = self.raw_dataset.df.loc[index, self.raw_dataset.one_hot_cols(self.input_cols)]
        return input_vec, output_vec

    def __len__(self):
        return self.raw_dataset.df.shape[0]


class DataSet():
    
    def __init__(self, data_frame_list, infer_categoricals=True, categorical_cols=None):
        """categorical_cols: list of categorical column _names_ """

        self.raw_df = pd.concat(data_frame_list, axis=0, ignore_index=True, sort=True)
        self.df = self.raw_df.copy()
        self.variable_names = list(self.df.columns)
 
        if not infer_categoricals:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = self.__sniff_categorical()
        
        self.variable_dict = dict()
        for v in self.variable_names:
            self.variable_dict[v] = dict()
            if v in self.categorical_cols:
                self.variable_dict[v]['type'] = "Categorical"
                le = preprocessing.LabelEncoder()
                inds = self.df[v].isnull() == False
                le.fit(self.df[v][inds])
                self.variable_dict[v]['dim'] = np.count_nonzero(le.classes_)
                self.variable_dict[v]['transform'] = le.transform
                self.variable_dict[v]['inverse_transform'] = le.inverse_transform
            else:
                self.variable_dict[v]['type'] = "Numeric"
                self.variable_dict[v]['dim'] = 1
                scaler = preprocessing.StandardScaler()
                scaler.fit(self.df[v].values.reshape([-1, 1]))
                self.variable_dict[v]['transform'] = scaler.transform
                self.variable_dict[v]['inverse_transform'] = scaler.inverse_transform

        for v in self.variable_names:
            if(self.variable_dict[v]['type'] == 'Categorical'):
                non_missing_inds = self.df[v].isnull() == False
                self.df[v][non_missing_inds] = self.variable_dict[v]['transform'](self.df[v][non_missing_inds])
                self.df[v].astype('category')
            else:
                self.df[v] = self.variable_dict[v]['transform'](self.df[v].values.reshape([-1, 1]))

    def __sniff_categorical(self):
        num_unique_values = [len(self.df[c].unique()) for c in list(self.df)]
        categorical_cols = [list(self.df)[i]
                            for i in range(len(num_unique_values))
                            if num_unique_values[i] < max(np.log(self.df.shape[0]), 5)]
        return categorical_cols

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