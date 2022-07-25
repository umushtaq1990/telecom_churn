import os,sys
import numpy as np
import pandas as pd
import pickle


sys.path.append("..") # Adds higher directory to python modules path.
curdir = os.path.dirname(__file__)
os.chdir(curdir)

def Get_Data(FILE_PATH):

    return df_pred

def Get_Pred(DF, MODEL_FILE):
    load_model = pickle.load(open(f'model/{MODEL_FILE}', 'rb')) 
    pred_prob = load_model.predict_proba(DF)[:, 1] 
    df_pred = pd.DataFrame({f'Prob': pred_prob*100})
    df_pred['ID'] = DF.index
    df_pred = df_pred.astype({f'Prob':'int64'})
    return df_pred


if __name__ == '__main__':
    df_pred = Get_Pred(df_data, 'RF_V001.pkl')
