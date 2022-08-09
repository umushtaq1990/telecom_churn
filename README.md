# Telecom Churn Analysis

This project could be divided into four parts:
- Exploratray data analysis (EDA) results available in Adhoc_Analysis.ipynb file
- Customer segments analysis using KMEANS, PCA, TSNE to get to know more about user groups.
- Machine learning models training and evaluation done in ML_Model_Train.ipynb file
- Falsk API deployment on Microsoft Azure / Code checkedin on github

As far as directory structure is concerned *data* folder contains validation and training datasets. *model* folder contains RF model as pickle file.*img* folder contains deployment results picture from Azure.
*Adhoc_Analysis.ipynb* contain EDA results till feature engineering
*ClusterAnalysis.ipynb* contains cluster or segments analysis for customers
*ML_Model_Train.ipynb* contains Machine Learning model generation, tuning results
*app.py* contains flask app module to get churn prediction results for given customer ID.
*get_pred.py* is used to get predictions once model is available.
*utilties.py* contains utilty functions.
*config.ini* contains configuration parameters, e.g: paths, numeric variables scaling
*requirements.txt* contains all packages required to build image
*Dockerfile* contain all steps to generate docker image. 
*unit_tests.py* contains all unittest, this need to be run in continuous integration and continuous deployment pipeline to check if realse is stable

## EDA
In this part numeric and categorical features distribution is analysed and statistical tests are used to check each feature
dependence with respect to target variable 'churn'. Results are available in *Adhoc_Analysis.ipynb* file and could be viewed in presentation format. 
Note : utilties.py file is required for this module where all necessary functions are available

## Model Training and Evaluation
Here different models are used to predict churn customers. After looking at intial results RF model is selcted for further hyperparameter tuning to get better results. RF Model is saved as pickle file under model folder once satisfing results are available.
These steps are performed in *ML_Model_Train.ipynb* file.

## Model deployment
Model is deployed as flask web application on Microsoft Azure. Intially all code is commited to github *https://github.com/umushtaq1990/telecom_churn* , than it is deployed to cloud enviornment. *img* folder contains the pictures showing deployment status on Azure.


## Docker Image
Build docker image: 
- docker build -t telecom_churn:v003 -f Dockerfile .
Run docker image: 
- docker run -d -p 9090:9090 telecom_churn:v003