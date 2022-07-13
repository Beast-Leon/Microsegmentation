#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:49:03 2022

@author: leon
"""
import numpy as np
import pandas as pd


# Maximum "N" one hot columns, the rest should merge to be one column named as "others"
# The logics is somewhat wrong. Here we keep at most N columns for all the categorical columns
# What we want is for each categorical column, we want at most N one hot columns.
def convert_onehot_v1(df, onehot_encoder, N = 5):
    """
    input a dataframe and an onehot encoder, generating onehot-encoded feature
    data. Set the total maximum onehot columns to be 5.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    onehot_encoder : TYPE
        DESCRIPTION.
    N : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    encoded_array : TYPE
        DESCRIPTION.
    encoded_name : TYPE
        DESCRIPTION.

    """
    encoded_array = onehot_encoder.fit_transform(df).toarray()
    features = onehot_encoder.feature_names_in_

    categories = onehot_encoder.categories_

    num_cols = encoded_array.shape[1]
    
    encoded_name = []
    for cat, col in zip(categories, features):
        for each_cat in cat:
            encoded_name.append(each_cat + "_" + col)
    top_indices = []
    if N > num_cols:
        print("Your desired maximum number of one hot columns is not valid, setting to be the current maximum number of one hot columns.")
        N = num_cols
    if num_cols > N:
        # select columns with top max_cols counts of values, the rest named as others
        count_sum = encoded_array.sum(axis = 0)
        indices = sorted(range(len(count_sum)), key=lambda i: count_sum[i])
        top_indices = indices[-N:]

        rest_indices = indices[:-N]

        selected_cols = encoded_array[:, top_indices]
        rest_col = np.sum(encoded_array[:, rest_indices], axis = 1)
        rest_col = np.expand_dims(rest_col, axis = 1)
        encoded_array = np.append(selected_cols, rest_col, axis = 1)
    
        
        encoded_name = [encoded_name[i] for i in top_indices]
        encoded_name.append('Others')
    
    return encoded_array, encoded_name
# Preprocess the input dataframe, return X, y, column name list, and column types
def preprocess(df,
              ordinal_encoder,
              onehot_encoder,
              random_state = 0,
              N = 10):
    df = df.copy(deep = True)
    column_type = df.dtypes
    categorical_columns = []
    
    for index, name in enumerate(df.columns):
        cur_type = column_type[index]
        if cur_type != 'int64' and cur_type != 'float64':
            if index != len(df.columns) - 1:
                categorical_columns.append(name)
            else:
                df[name] = ordinal_encoder.fit_transform(df[[name]]).astype('int64')
    # Extract target variable
    y = df[df.columns[-1]].to_numpy()
    # Convert categorical columns to one hot columns
    encoded_array, encoded_name = convert_onehot_v1(df[categorical_columns], onehot_encoder, N = N)
    
    encoded_df = pd.DataFrame(data = encoded_array, columns = encoded_name).astype('object')
    
    # Drop previous categorical columns and target variable
    df.drop(df.columns[-1], axis = 1, inplace = True)
    df.drop(categorical_columns, axis = 1, inplace = True)
    
    # Concat the old and new data frames
    new_df = pd.concat([df, encoded_df], axis = 1)
    X = new_df.to_numpy()
    target_names = ordinal_encoder.categories_
    return X, y, np.array(new_df.columns), new_df.dtypes, target_names







