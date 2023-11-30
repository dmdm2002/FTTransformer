import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from impyute.imputation.cs import mice
from impyute.imputation.cs import fast_knn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# cols = ['A_Sex', 'Op_AN', 'Op_Type', 'Type_Adm', 'B_AKI', 'B_CAD',
#         'B_CKD', 'B_COPD', 'B_CVD', 'B_Malig', 'AB_DM', 'AN_Asthma',
#        'AN_COPD', 'AN_Heart_Dz', 'AN_Liver_Dz', 'AN_Vascular_Dz', 'A_Age',
#         'AB_HTN', 'A_DBP', 'A_SBP',
#         'O_AKI', 'O_Critical_AKI_90', 'O_Death_90', 'O_RRT_90']
class Preprocessor(object):
    def __init__(self):
        super().__init__()
        self.codeboock = pd.read_excel('D:/Side/MAIC/SyntheticAKI/Codebook.xlsx')
        nuumerical_cols = self.codeboock['Nuumerical variables'].dropna().to_list()

        temp = ['B_PTH', 'B_Triglyceride', 'B_HbA1c', 'A_HR', 'A_WT', 'B_tCO2', 'B_ESR', 'B_LDL', 'A_HT', 'B_HDL', 'B_UPCR']

        self.numerical = []
        for col in nuumerical_cols:
            if col not in temp:
                self.numerical.append(col)

    def get_categorical_cols(self):
        return self.codeboock['Categorical variables'].values

    def del_null(self, train, test):
        print('-------[너무 많은 결측값을 가진 칼럼을 제거]-------')
        null_count = train.isna().sum()
        null_count = null_count.reset_index().rename(columns={'index': 'id', 0: 'val'})
        drop_labels = null_count.loc[(null_count['val'] >= 160000)]['id'].to_list()
        train.drop(labels=drop_labels, axis=1, inplace=True)
        test.drop(labels=drop_labels, axis=1, inplace=True)

        return train, test

    def make_A_HTN(self, df):
        print('-------[고혈압/저혈압/정상 칼럼 생성]-------')
        df.loc[(df['A_SBP'] < 140) | (df['A_DBP'] < 90), 'A_HTN'] = 0
        df.loc[(df['A_SBP'] >= 140) | (df['A_DBP'] >= 90), 'A_HTN'] = 1
        df.loc[(df['A_SBP'] <= 90) | (df['A_DBP'] <= 60), 'A_HTN'] = 2

        return df

    def std_scaler(self, train, test):
        print('-------[Standard scaler 실행]-------')
        std = StandardScaler()
        nuumerical_tr = train[self.numerical]
        nuumerical_te = test[self.numerical]

        tr = train.drop(labels=self.numerical, axis=1)
        te = test.drop(labels=self.numerical, axis=1)

        tr.fillna(0, inplace=True)
        te.fillna(0, inplace=True)

        std.fit(nuumerical_tr)

        tr_std = std.transform(nuumerical_tr)
        te_std = std.transform(nuumerical_te)

        print('-------[Start FastKNN 결측값 채워 넣기]-------')
        # knn_imputer = KNNImputer(n_neighbors=5)
        # #  KNN을 사용해 numerical null 값 대체 --> 변수의 유사도를 사용하여 채워진다.
        # print('Train DATA!')
        # np_imputed_tr = knn_imputer.fit_transform(tr_std)
        # print('Test DATA!')
        # np_imputed_te = knn_imputer.transform(te_std)

        tr_std = pd.DataFrame(tr_std, columns=self.numerical)
        te_std = pd.DataFrame(te_std, columns=self.numerical)

        tr_std.fillna(0, inplace=True)
        te_std.fillna(0, inplace=True)

        # tr_concat = pd.concat([tr, tr_std], axis=1)
        # te_concat = pd.concat([te, te_std], axis=1)

        return tr, te, tr_std, te_std

    def cut_3sigma(self, tr_X_cate, tr_X_num, tr_y):
        print('-------[이상치 제거!!]-------')
        for i in range(len(self.numerical)):
            column_data = tr_X_num.iloc[:, i]
            quan_25 = np.percentile(column_data.values, 25)
            quan_75 = np.percentile(column_data.values, 75)

            iqr = quan_75 - quan_25
            iqr = iqr * 1.5
            lowest = quan_25 - iqr
            highest = quan_75 + iqr
            outlier_index = column_data[(column_data < lowest) | (column_data > highest)].index

            tr_X_num.drop(outlier_index, axis=0, inplace=True)
            tr_X_cate.drop(outlier_index, axis=0, inplace=True)
            tr_y.drop(outlier_index, axis=0, inplace=True)

        return tr_X_num, tr_X_cate, tr_y
