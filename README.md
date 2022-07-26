# Telecom Churn Analysis

This project could be divided into three parts:
- Exploratray data analysis (EDA) results available in Adhoc_Analysis.ipynb file
- Machine learning models training and evaluation done in Adhoc_Analysis.ipynb file
- Falsk API deployment on Microsoft Azure / Code checkedin on github

As far as directory structure is concerned *data* folder contains validation and training datasets. *model* folder contains RF model as pickle file. 
In *Adhoc_Analysis.ipynb* all steps from EDA till Machine Learning model generation are performed.
*app.py* contains flask app module to get churn prediction results for given customer ID.
*get_pred.py* is used to get predictions once model is available.
*utilties.py* contains utilty functions.
*config.ini* contains configuration parameters, e.g: paths, numeric variables scaling
*requirements.txt* contains all packages required to build image
*Dockerfile* contain all steps to generate docker image. 

## EDA
In this part numeric and categorical features distribution is analysed and statistical tests are used to check each feature
dependence with respect to target variable 'churn'. Results are available in *Adhoc_Analysis.ipynb* file and could be viewed in presentation format. 
Note : utilties.py file is required for this module where all necessary functions are available

## Model Training and Evaluation
Here different models are used to predict churn customers. After looking at intial results RF model is selcted for further hyperparameter tuning to get better results. RF Model is saved as pickle file under model folder once satisfing results are available.
These steps are performed in *Adhoc_Analysis.ipynb* file.

## ML Model deployment
Model is deployed as flask web application on Microsoft Azure. Intially all code is commited to github *https://github.com/umushtaq1990/telecom_churn* , than it is deployed to cloud enviornment. 

