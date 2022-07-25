import os,sys
import numpy as np
import pandas as pd
import pickle
from utilties import *

sys.path.append("..") # Adds higher directory to python modules path.
curdir = os.path.dirname(__file__)
os.chdir(curdir)

def Get_Data(ID, FILE_PATH):
    df = pd.read_pickle(FILE_PATH)
    df = df.loc[df.customerID==ID]
    print('df',df)
    cat_feats = ['gender', 'SeniorCitizen', 'Product: International', 'Product: Voice mail','Phone Code','PaperlessBilling','service calls','churn'] #categorical
    num_feats = list(set(df.columns)-set(['customerID','Telephone Number', 'US State']+cat_feats))   # get numeric feats
    num_feats = list(set(num_feats)-set(['Total, EUR', 'eve EUR', 'night EUR', 'internatonal EUR']))   # remove highly correlated numeric feats
    df = df.set_index('customerID')
    df = df[num_feats+cat_feats]
    df = Convert_Cat_Feats(df)
    df = Scale_Num_Feats_Pred(df, num_feats)
    return df

def Get_Pred(DF, MODEL_FILE):
    load_model = pickle.load(open(f'{MODEL_FILE}', 'rb')) 
    pred_prob = load_model.predict_proba(DF)[:, 1] 
    df_pred = pd.DataFrame({f'Per_Prob': pred_prob*100})
    df_pred['ID'] = DF.index
    df_pred = df_pred.astype({f'Per_Prob':'int64'})
    return df_pred[['ID','Per_Prob']]

def Main_Pred(ID):
    df_data = Get_Data(ID, 'data/df_validation.pkl')
    x_data = df_data.drop(columns='churn')
    y_data = df_data.loc[:, 'churn']
    print(x_data)
    df_pred = Get_Pred(x_data, 'model/RF_V001.pkl')
    return df_pred

if __name__ == '__main__':
    Main_Pred('8022-BECSI')
