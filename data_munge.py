import os

import numpy as np
import pandas as pd
import sys
from statsmodels.imputation import mice

# suppress printing
#sys.stdout = open(os.devnull, 'w')

# data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'data')

# dictionaries created for simplicity in managing file paths
household_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'A_hhold_test.csv')},

                   'B': {'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'B_hhold_test.csv')},

                   'C': {'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'),
                         'test': os.path.join(DATA_DIR, 'C_hhold_test.csv')}}


individual_paths = {'A': {'train': os.path.join(DATA_DIR, 'A_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'A_indiv_test.csv')},

                    'B': {'train': os.path.join(DATA_DIR, 'B_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'B_indiv_test.csv')},

                    'C': {'train': os.path.join(DATA_DIR, 'C_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'C_indiv_test.csv')}}

# read data in
a_h_train = pd.read_csv(household_paths['A']['train'], index_col='id')
b_h_train = pd.read_csv(household_paths['B']['train'], index_col='id')
c_h_train = pd.read_csv(household_paths['C']['train'], index_col='id')

a_i_train = pd.read_csv(individual_paths['A']['train'], index_col=['id', 'iid'])
b_i_train = pd.read_csv(individual_paths['B']['train'], index_col=['id', 'iid'])
c_i_train = pd.read_csv(individual_paths['C']['train'], index_col=['id', 'iid'])

# read test data in
a_h_test = pd.read_csv(household_paths['A']['test'], index_col='id')
b_h_test = pd.read_csv(household_paths['B']['test'], index_col='id')
c_h_test = pd.read_csv(household_paths['C']['test'], index_col='id')

a_i_test = pd.read_csv(individual_paths['A']['test'], index_col=['id', 'iid'])
b_i_test = pd.read_csv(individual_paths['B']['test'], index_col=['id', 'iid'])
c_i_test = pd.read_csv(individual_paths['C']['test'], index_col=['id', 'iid'])

def standardize(df, numeric_only=True):
    # detect columns that are numeric
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))


    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))


    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    #df.fillna(0, inplace=True)

    return df

def impute(df, type='household', perturbation_method='gaussian', k_pmm=20, history_callback=None):
    # wrapper for impute to preserve index

    if type is 'household':
        index = df.index

    imputed_df = mice.MICEData(df, perturbation_method, k_pmm, history_callback).data

    imputed_df.set_index(index, inplace=True)
    return imputed_df


def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)

    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


aX_h_train = pre_process_data(a_h_train.drop('poor', axis=1))
ay_h_train = np.ravel(a_h_train.poor)

bX_h_train = pre_process_data(b_h_train.drop('poor', axis=1))
by_h_train = np.ravel(b_h_train.poor)

cX_h_train = pre_process_data(c_h_train.drop('poor', axis=1))
cy_h_train = np.ravel(c_h_train.poor)

aX_i_train = pre_process_data(a_i_train.drop('poor', axis=1))
ay_i_train = np.ravel(a_i_train.poor)

bX_i_train = pre_process_data(b_i_train.drop('poor', axis=1))
by_i_train = np.ravel(b_i_train.poor)

cX_i_train = pre_process_data(c_i_train.drop('poor', axis=1))
cy_i_train = np.ravel(c_i_train.poor)

a_h_test = pre_process_data(a_h_test, enforce_cols=aX_h_train.columns)
b_h_test = pre_process_data(b_h_test, enforce_cols=bX_h_train.columns)
c_h_test = pre_process_data(c_h_test, enforce_cols=cX_h_train.columns)

a_i_test = pre_process_data(a_i_test, enforce_cols=aX_i_train.columns)
b_i_test = pre_process_data(b_i_test, enforce_cols=bX_i_train.columns)
c_i_test = pre_process_data(c_i_test, enforce_cols=cX_i_train.columns)