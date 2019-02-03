from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import re


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


class RawData():
    def __init__(self, csv_path, infer_categoricals=True, categorical_cols=None):
        """categorical_cols: list of categorical column _names_ """

        self.df = pd.read_csv(csv_path)
        self.col_info = dict(zip(list(self.df), [["Numeric", None]] * self.df.shape[1]))

        if not infer_categoricals:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = self.__sniff_categorical()

        for col in self.categorical_cols:
            self.df[col] = self.df[col].astype('category')
            self.col_info[col] = ['Categorical', dict(enumerate(self.df[col].cat.categories))]
            self.df[col] = self.df[col].cat.codes  # convert to integer representation. self.col_info will give mapping
            self.df[col] = self.df[col].astype('category')

        self.df_one_hot = pd.get_dummies(self.df)
        self.one_hot_col_map = {}
        for col in list(self.df):
            r = re.compile(col)
            self.one_hot_col_map[col] = list(filter(r.match, list(self.df_one_hot)))

    def __sniff_categorical(self):

        num_unique_values = [len(self.df[c].unique()) for c in list(self.df)]
        categorical_cols = [list(self.df)[i]
                            for i in range(len(num_unique_values))
                            if num_unique_values[i] < min(np.sqrt(self.df.shape[0]), 10)]
        return categorical_cols

    def one_hot_cols(self, col_names):
        one_hot_cols = sum([self.one_hot_col_map[c] for c in col_names], [])
        return one_hot_cols

    pass



if __name__ == '__main__':
    r = RawData("./data/5d.csv")
    print(r.col_info)
    print(r.df.describe(include="all"))
    print(r.df.iloc[0])
    print("-----------")
    print(r.df.dtypes)
    print("............")
    print(pd.get_dummies(r.df.iloc[0]))
    print(">>>>>>>>>>>")
    print(pd.get_dummies(r.df).iloc[0])
    print(">>>>>>>>>>>")
    print(r.df_one_hot.dtypes)