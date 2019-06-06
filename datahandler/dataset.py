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


class DataUtils():
    
    def __init__(self, data_frame_list, infer_categoricals=True, categorical_cols=None):
        """categorical_cols: list of categorical column _names_ """

        self.df = pd.concat(data_frame_list, axis=0, ignore_index=True)
        self.variables = list(self.df.columns)
        self.col_info = dict(zip(list(self.df), [["Numeric", None]] * self.df.shape[1]))

        if not infer_categoricals:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = self.__sniff_categorical()
        
        self.label_encoders = dict()
        for v in self.categorical_cols:
            self.label_encoders[v] = preprocessing.LabelEncoder()
            self.label_encoders[v].fit(self.df[v])

    def __sniff_categorical(self):

        num_unique_values = [len(self.df[c].unique()) for c in list(self.df)]
        categorical_cols = [list(self.df)[i]
                            for i in range(len(num_unique_values))
                            if num_unique_values[i] < max(np.sqrt(self.df.shape[0]), 5)]
        return categorical_cols

    pass



if __name__ == '__main__':
    r = DataUtils("./data/5d.csv")
    print(r.col_info)
    print(r.df.describe(include="all"))
    print(r.df.iloc[0])
    print("-----------")
    print(r.df.dtypes)
    print("............")
    print(pd.get_dummies(r.df.iloc[0]))
    print(">>>>>>>>>>>")
    print(pd.get_dummies(r.df).iloc[0])
