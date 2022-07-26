import numpy as np
import pandas as pd
import math
from decimal import Decimal
from configparser import ConfigParser
from scipy.stats import chi2_contingency
from scipy import stats
from sklearn.metrics import mutual_info_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, classification_report,confusion_matrix,roc_auc_score,roc_curve
import pickle


def Get_Summary_DF(DF):
    ds_cat_stats = pd.DataFrame(columns = ['column', 'values', 'values_count_incna', 'values_count_nona', 'num_miss', 'pct_miss'])
    tmp = pd.DataFrame()

    for c in DF.columns:
        tmp['column'] = [c]
        tmp['values'] = [DF[c].unique()]
        tmp['values_count_incna'] = len(list(DF[c].unique()))
        tmp['values_count_nona'] = int(DF[c].nunique())
        tmp['num_miss'] = DF[c].isnull().sum()
        tmp['pct_miss'] = (DF[c].isnull().sum()/ len(DF)).round(3)*100
        ds_cat_stats = ds_cat_stats.append(tmp)
    return ds_cat_stats

def Plot_Bar_Mit_Num_Per(DF, COLUMN_NAME, AXIS, TITLE):
    DF[COLUMN_NAME].value_counts().plot(kind='bar',title=TITLE, ax = AXIS)
    for p in AXIS.patches:
        AXIS.annotate(str(p.get_height())+'  ->  '+str(round(Decimal(p.get_height()/DF.shape[0]),2)), (p.get_x(), p.get_height()))

def Per_Stacked_Bar_Plot_Cat_Feats(DF, COLS_TO_PLOT, MAIN_TITLE):
    number_of_columns = 2
    number_of_rows = math.ceil(len(COLS_TO_PLOT)/2)
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(MAIN_TITLE, fontsize=22,  y=.95)
    for index, column in enumerate(COLS_TO_PLOT, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        prop_by_independent = pd.crosstab(DF[column], DF['churn']).apply(lambda x: x/x.sum()*100, axis=1)
        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,rot=0, color=['navy','salmon'])
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5), title='Churn', fancybox=True)
        ax.tick_params(rotation='auto')
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

def Plot_Bar_Mit_Num_Per(DF, COLUMN_NAME, AXIS, TITLE):
    DF[COLUMN_NAME].value_counts().plot(kind='bar',title=TITLE, ax = AXIS)
    for p in AXIS.patches:
        AXIS.annotate(str(p.get_height())+'  ->  '+str(round(Decimal(p.get_height()/DF.shape[0]),2)), (p.get_x(), p.get_height()))

def Plot_Bar_Mit_Num(DF, COLUMN_NAME, AXIS, TITLE):
    DF[COLUMN_NAME].value_counts().plot(kind='bar',title=TITLE, ax = AXIS)
    for p in AXIS.patches:
        AXIS.annotate(p.get_height(), (p.get_x(), p.get_height()))

def Per_Stacked_Bar_Plot(DF, FEAT,TARGET_FEAT, AXIS, TITLE):
    prop_by_independent = pd.crosstab(DF[FEAT], DF[TARGET_FEAT]).apply(lambda x: x/x.sum()*100, axis=1)
    prop_by_independent.plot(kind='bar', ax=AXIS, stacked=True, title=TITLE, color=['navy','salmon'])
    AXIS.xaxis.label.set_visible(False)
    AXIS.legend(loc="upper right")
            
def Test_ANOVA_Cat_Num(DF, FEAT,TARGET_FEAT):
    Conversion_0 = DF[DF[TARGET_FEAT]==0]
    Conversion_1 = DF[DF[TARGET_FEAT]==1]
    F, p = stats.f_oneway(Conversion_0[FEAT],  Conversion_1[FEAT])
    #print('F Statistic:', F, '\tp-value:', p)
    if p>0.05:
        print(f'[TARGET_FEAT] and [{FEAT}] are Independent with Chi-sq pvalue:{str(round(Decimal(p),3))}')
    else:
        print(f'[TARGET_FEAT] and [{FEAT}] are dependent with Chi-sq pvalue:{str(round(Decimal(p),3))}')
        
def Test_ANOVA_Cat(DF, COL1, COL2):
    table = pd.crosstab(DF[COL1],DF[COL2])
    stat, pvalue, dof, expected = chi2_contingency(table)
    if pvalue>=0.05:
        print(f'[{COL1}] and [{COL2}] are independent with Chi-sq pvalue:{str(round(Decimal(pvalue),3))}') #accept Ho
    else:
        print(f'[{COL1}] and [{COL2}] are dependent with Chi-sq pvalue:{str(round(Decimal(pvalue),3))}')  # reject Ho
        
def Plot_StackedBar_Mit_Num(DF, COLUMN1, COLUMN2, AXIS, TITLE):
    Test_ANOVA_Cat(DF, COLUMN1, COLUMN2)
    pd.crosstab(DF[COLUMN1], DF[COLUMN2]).plot(kind='bar',stacked=False, title=TITLE, ax = AXIS)
    AXIS.xaxis.label.set_visible(False)
    for p in AXIS.patches:
        AXIS.annotate(p.get_height(), (p.get_x(), p.get_height()))

def Plot_Cat_Col(DF, FEAT,TARGET_FEAT):
    fig, (axes) = plt.subplots(1,3, figsize=(12,5))
    fig.subplots_adjust(hspace=0.35)
    Plot_Bar_Mit_Num(DF , FEAT, axes[0], FEAT)
    Plot_StackedBar_Mit_Num(DF,FEAT, TARGET_FEAT, axes[1], f"{FEAT} vs {TARGET_FEAT}")
    Per_Stacked_Bar_Plot(DF, FEAT,TARGET_FEAT, axes[2], f"{FEAT} vs {TARGET_FEAT} (PER)")

def Numeric_Distribution_Plot(DF, FEAT, TARGET_FEAT):
    Test_ANOVA_Cat_Num(DF, FEAT,TARGET_FEAT)
    COL = FEAT  #'wc_name'
    feature = list(DF['churn'].unique())

    f, (ax) = plt.subplots(2, 2, figsize=(12,8 ))
    f.suptitle(f'{COL} Trend vs {TARGET_FEAT}', fontsize=14)
    f.subplots_adjust(top=0.90, wspace=0.3)

    sns.boxplot( y=COL, data=DF, ax=ax[0,0])
    ax[0,0].set_ylabel(COL,size = 12,alpha=0.8)
    
    sns.distplot(DF[COL], hist = True, kde = True, kde_kws = {'linewidth': 3}, ax=ax[0,1])
    ax[0,1].set_xlabel('',size = 12,alpha=0.8)
    ax[0,1].set_ylabel("Density",size = 12,alpha=0.8)
    
    sns.boxplot(x=TARGET_FEAT, y=COL, data=DF, ax=ax[1,0])
    ax[1,0].set_xlabel(TARGET_FEAT,size = 12,alpha=0.8)
    ax[1,0].set_ylabel(COL,size = 12,alpha=0.8)

    for case in feature:
        subset = DF[DF[TARGET_FEAT] == case]
        if subset.shape[0]>1:
            sns.distplot(subset[COL], hist = False, kde = True, kde_kws = {'linewidth': 3}, label = case, ax=ax[1,1])

    ax[1,1].set_xlabel(COL,size = 12,alpha=0.8)
    ax[1,1].set_ylabel("Density",size = 12,alpha=0.8)

def Convert_Cat_Feats(DF):
    DF['Product: International'] = DF['Product: International'].map({'yes':1, 'no':0})
    DF['Product: Voice mail'] = DF['Product: Voice mail'].map({'yes':1,'no':0})
    DF['PaperlessBilling'] = DF['PaperlessBilling'].map({'Yes':1,'No':0})
    DF['gender'] = DF['gender'].map({'Female':0,'Male':1})
    DF['Phone Code'] = DF['Phone Code'].map({408:0,415:1,510:2})
    DF['churn'] = DF['churn'].map({False:0,True:1})
    return DF

def ADD_PATH_CONFIG(CONFIG_OBJ):
    path_dict = {'model' : 'model/RF_V001.pkl', 
                'data' : 'data/df_validation.pkl', 
                'feats':['vmail', 'international calls', 'total day calls', 'Call day minutes','international minutes', 
                'Duration', 'eve calls', 'night minutes','night calls', 'eve minutes', 'gender', 'SeniorCitizen',
                'Product: International', 'Product: Voice mail', 'Phone Code','PaperlessBilling', 'service calls', 'churn']}
    CONFIG_OBJ['PATH'] = path_dict
    return CONFIG_OBJ

def Scale_Num_Feats_Train(DF, COLS_LIST):       
    config_object = ConfigParser()             # store scaling values of each feat in confg file to later used for pred on unseen data
    num_dict = {}
    for column in COLS_LIST:
        min_column = DF[column].min()
        max_column = DF[column].max()
        DF[column] = (DF[column] - min_column) / (max_column - min_column)
        num_dict.update({column:[min_column,max_column]}) 
    ADD_PATH_CONFIG(config_object)
    config_object['NUM_FEATS'] = num_dict
    with open('config.ini', 'w') as conf:
        config_object.write(conf)
    return DF

def Scale_Num_Feats_Pred(DF, COLS_LIST):              # scaling for each feat to be read from config file where values were stored during training
    config_object = ConfigParser()
    config_object.read("config.ini")
    num_feats_scale = config_object["NUM_FEATS"]
    for column in COLS_LIST:
        min_column = int(float(num_feats_scale[column].replace("[",'').replace("]",'').split(',')[0]))
        max_column = int(float(num_feats_scale[column].replace("[",'').replace("]",'').split(',')[1]))
        DF[column] = (DF[column] - min_column) / (max_column - min_column)
    return DF

def create_models(seed=2):
    models = []
    models.append(('dummy_classifier', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('k_nearest_neighbors', KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('support_vector_machines', SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    return models

def Get_ROC(MODEL, XTEST, YTEST):
    y_pred_default = MODEL.predict(XTEST)
    logit_roc_auc = metrics.roc_auc_score(YTEST, y_pred_default )
    fpr, tpr, thresholds = metrics.roc_curve(YTEST, y_pred_default )
    plt.figure()
    plt.plot(fpr, tpr, label='RFC Default (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print("Area under curve is:", round(metrics.roc_auc_score(YTEST, y_pred_default),2))
    print("Recall for our model is:" , round(metrics.recall_score(YTEST, y_pred_default),2))
    print("Accuracy on test set is:" , round(metrics.accuracy_score(YTEST, y_pred_default),2))
    print(confusion_matrix(YTEST,y_pred_default))
    print(classification_report(YTEST,y_pred_default))
    
def Tune_Single_Parameter(XTRAIN, YTRAIN, PAR_NAME, PARAMETER, N_FOLD, TARGET):
    print(f'\nTuning {PAR_NAME}')
    rf = RandomForestClassifier(class_weight = 'balanced',random_state=0)
    rf = GridSearchCV(rf, PARAMETER, return_train_score=True, cv=N_FOLD, n_jobs=-1, scoring=TARGET,verbose = 1)
    rf.fit(XTRAIN, YTRAIN)
    scores = rf.cv_results_
    print(rf.best_score_)
    print(rf.best_params_)
    plt.figure(figsize=(8,8))
    plt.plot(scores[f"param_{PAR_NAME}"], 
             scores["mean_train_score"], 
             label=f"training {TARGET}")
    plt.plot(scores[f"param_{PAR_NAME}"], 
             scores["mean_test_score"], 
             label=f"test {TARGET}")
    plt.xlabel(PAR_NAME)
    plt.ylabel(TARGET)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def Get_RF_Model_Feat_Importance(MODEL, FEATS, NUM_FEAT):
    coefs = MODEL.feature_importances_
    indices = np.argsort(coefs)[::-1]
    plt.figure(figsize=(16,6))
    plt.title("Feature importances (Random Forests)")
    plt.bar(range(NUM_FEAT), coefs[indices[:NUM_FEAT]], color="b", align="center")
    plt.xticks(range(NUM_FEAT), FEATS[indices[:NUM_FEAT]], rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.ion()
    plt.show()