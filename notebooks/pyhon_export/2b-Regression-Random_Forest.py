#!/usr/bin/env python
# coding: utf-8

# # Regression of pH range using machine learning (Random Forest)

# In[1]:


HOME_DIR = '/Users/datatranslators/Documents/GitHub/acido_CEST_Machine_Learning/'
DESTINATION = 'outputs'


# ## Libraries

# In[2]:


#libs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn import preprocessing as pp
import seaborn as sns

from sklearn.feature_selection import SelectFromModel

from joblib import dump, load


# ## Data

# In[3]:


# load data
acidoCEST_ML = pd.read_parquet('../clean_data/acido_CEST_MRI_Iopamidol.parquet.gzip')

# drop columns we cannot measure or that are redundant
cols_2_drop = ['ApproT1(sec)','Temp','ExpB0(ppm)','FILE','Conc(mM)']
acidoCEST_ML = acidoCEST_ML.drop(cols_2_drop, axis = 1)

# define experimental columns -- > things we measure other than CEST
exper_cols = [ 'ExpB1(percent)', 'ExpB0(Hz)', 'SatPower(uT)',  'SatTime(ms)','ExpT1(ms)', 'ExpT2(ms)']

# used during development
#acidoCEST_ML  = acidoCEST_ML.sample( n = 5_00)

acidoCEST_ML.shape


# ## Functions

# ### Random Forest

# In[4]:


# Regression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor as RFR

def train_RFR(Xdata, pH_observed):
    print('------------------ Random Forest ------------------ ')
    print()
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(Xdata, pH_observed, test_size=0.30, random_state=42)
        
        
    
    max_f = int( 0.5 * Xdata.shape[1] )
        
    # Regression
    #param_grid = dict( n_estimators = [100,200,500], max_depth =[None], max_features=['sqrt','log2','auto',max_f], max_samples=[.10,.50], min_samples_leaf=[.01,.50] )

    
    param_grid = dict( n_estimators = [100,200,500], max_depth =[10,20,40,None] )

    
    # grid parameters
    scorer = metrics.make_scorer( metrics.r2_score, greater_is_better = True) 
    
    #
    estimator = GridSearchCV( RFR(random_state = 42,  n_jobs = -1), param_grid, verbose = 3, cv = 3, n_jobs= 1, scoring=scorer )

    # fit
    estimator.fit(X_train, y_train)

    score_train = np.round( 100 * metrics.mean_absolute_percentage_error(y_train , estimator.predict(X_train)), 2)
    score_test  = np.round( 100 * metrics.mean_absolute_percentage_error(y_test , estimator.predict(X_test)), 2)

    #score on pH units
    score_pH_train = np.mean( np.abs(  y_train -  estimator.predict(X_train) ) )
    score_pH_test = np.mean( np.abs(  y_test -  estimator.predict(X_test) ) )


    m = f' mean absolute percentage error \n train = {score_train} % \n test  = {score_test} %'
    m2 = f'\n pH units |error : \ntrain = {score_pH_train:.3f} \n test = {score_pH_test:.3f}'
    
    #print(', '.join(Xdata.columns.to_list()))
    print()
    print(m,'\n',m2,'\n')
   
    plt.figure()
    sns.regplot(x = estimator.best_estimator_.predict(X_train), y =  y_train)
    plt.xlabel('\n Predicted pH')
    plt.ylabel('Measured pH \n')
    plt.title('Train Random Forest')
    
    plt.figure()
    sns.regplot(x = estimator.best_estimator_.predict(X_test), y =  y_test)
    plt.xlabel('\n Predicted pH')
    plt.ylabel('Measured pH \n')
    plt.title('Test Random Forest')
    
    print(estimator.best_estimator_)
    
    scores = dict()
    scores['mae_train'] = score_train
    scores['mae_test'] = score_test
    scores['pH_train'] = score_pH_train
    scores['pH_test'] = score_pH_test
    
    test_data = dict()
    test_data['X'] = X_test
    test_data['Y'] = y_test
    
    return estimator.best_estimator_, scores, test_data

def save_and_print(RFO, name='foo'):
    print('-----')
    for k in RFO.get_params().keys():
        print(f'{k}: {RFO.get_params()[k]}')
    
    dump(RFO, HOME_DIR + DESTINATION + f'/{name}.joblib') 


# # Results

# ## Random Forest

# ### 4.2 and 5.6 only

# In[5]:


get_ipython().run_cell_magic('time', '', "\nXdata = acidoCEST_ML[['4.2','5.6']]\nY     = acidoCEST_ML.pH\n\nreg_rf, _, _ = train_RFR(Xdata,Y)\n\nsave_and_print(reg_rf, name='RF_regressor_4256')\n")


# #### Feature Importance

# In[6]:


I = pd.DataFrame( 100 * reg_rf.feature_importances_, columns=['Importance'], index = Xdata.columns)
_, ax = plt.subplots(dpi = 100, figsize=(16,6))
I.plot(kind='bar', ax = ax, rot=90)
plt.title('Feature Importance for Regression of pH \n Only CEST features were used\n')


# ### Only CEST data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nXdata = acidoCEST_ML.drop(exper_cols + ['pH'],axis=1)\nYdata = acidoCEST_ML.pH \n\nreg_rf_only_CEST, _, _  = train_RFR(Xdata,Ydata)\nsave_and_print(reg_rf_only_CEST, name='RF_regressor_all_CEST')\n")


# #### Feature Importance

# In[ ]:


I = pd.DataFrame( 100 * reg_rf_only_CEST.feature_importances_, columns=['Importance'], index = Xdata.columns)
_, ax = plt.subplots(dpi = 100, figsize=(16,6))
I.plot(kind='bar', ax = ax, rot=90)
plt.title('Feature Importance for Regression of pH \n Only CEST features were used\n')


# ### All data

# In[ ]:


get_ipython().run_cell_magic('time', '', "Xdata = acidoCEST_ML.drop( ['pH'],axis=1) \nYdata = acidoCEST_ML.pH\n\nreg_rf_all, _, _  = train_RFR(Xdata,Ydata)\nsave_and_print(reg_rf_all, name='RF_regressor_all_DATA')\n")


# #### Feature Importance

# In[ ]:


I = pd.DataFrame( 100 * reg_rf_all.feature_importances_, columns=['Importance'], index = Xdata.columns)
_, ax = plt.subplots(dpi = 100, figsize=(16,6))
I.plot(kind='bar', ax = ax, rot=90)
plt.title('Feature Importance for Regression of pH \n All features were used\n')


# ### Recursive Feature elimination

# In[ ]:


get_ipython().run_cell_magic('time', '', "selected_RF = SelectFromModel( reg_rf_all  , max_features=20) \nselected_RF.fit(Xdata, Ydata)\ncols = Xdata.columns[selected_RF.get_support()]\n\nprint(cols)\n\nXdata = acidoCEST_ML[cols]\nYdata = acidoCEST_ML.pH\n\nreg_rf_selected_features, _, test_data_selected_features  = train_RFR(Xdata,Ydata)\n\nsave_and_print(reg_rf_selected_features, name='RF_regressor_selected')\n")


# #### Feature Importance

# In[ ]:


I = pd.DataFrame( 100 * reg_rf_selected_features.feature_importances_, columns=['Importance'], index = Xdata.columns)
_, ax = plt.subplots(dpi = 100, figsize=(16,6))
I.plot(kind='bar', ax = ax, rot=90)
plt.title('Feature Importance for Regression of pH \n All features from recursive elimination were used\n')


# ## Save predictions with selected

# In[ ]:


yhat = reg_rf_selected_features.predict( test_data_selected_features['X'] )

out = pd.DataFrame(yhat, columns=['Predicted_pH_RFReg_TEST'])
out['Measured_pH_TEST'] = test_data_selected_features['Y'].values
out.to_csv('../outputs/RF_reg_TEST_pred.csv')

out


# ## Remove B1 if selected an important feature

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ncols_without_B1 = [x for x in cols if 'B1' not in x ]\nprint(cols_without_B1)\n\nXdata = acidoCEST_ML[cols_without_B1]\nYdata = acidoCEST_ML.pH\n\nreg_rf_RFE_noB1, _, _  = train_RFR(Xdata,Ydata)\nsave_and_print(reg_rf_RFE_noB1, name='RF_regressor_selected_no_B1')\n")


# ## Effect of number of  estimators using Recursive Feature elimination features

# In[ ]:


def train_RFR_02(Xdata, pH_observed):
    print('------------------ Random Forest ------------------ ')
    print()
    # grid
    param_grid = dict( n_estimators = [1,2,5,10,20,40,80,160], max_depth = [2**(x) for x in [1,2,3,4,5,6]])

    
    # grid parameters
    scorer = metrics.make_scorer( metrics.r2_score, greater_is_better = True) 
    
    #
    estimator = GridSearchCV( RFR(random_state = 42,  n_jobs = -1), param_grid, verbose = 0, cv = 10, n_jobs= 1, scoring=scorer )

    # fit
    estimator.fit(Xdata, pH_observed)

    score_train = np.round( 100 * metrics.mean_absolute_percentage_error(pH_observed , estimator.predict(Xdata)), 2)
    score_test  = np.round( 100 * metrics.mean_absolute_percentage_error(pH_observed , estimator.predict(Xdata)), 2)



    m = f' mean absolute percentage error \n train = {score_train} % \n test  = {score_test} %'
    
    print(estimator.best_estimator_)
    
    return estimator.best_estimator_, estimator

reg, CV = train_RFR_02(Xdata, Ydata)

R = pd.DataFrame()

pars = ['param_max_depth', 'param_n_estimators','std_test_score','mean_test_score']

for p in pars:
    R[p] = CV.cv_results_[p]

R['param_max_depth'] = R['param_max_depth'].astype(str) 
    
plt.figure(dpi=100)

for n  in R.param_n_estimators.unique():
    scores = R[R.param_n_estimators == n]
    plt.errorbar(x=scores['param_max_depth'], y=scores['mean_test_score'], yerr=scores['std_test_score'])
    
plt.legend([f'{x} trees' for x in R.param_n_estimators.unique()])

plt.xlabel('\n Max Depth for each tree')

plt.ylabel('Mean R2 score \n')


# ## Matrix Regression

# In[ ]:


# Regression

def train_RFR_without_split(Xdata, pH_observed):
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(Xdata, pH_observed, test_size=0.30, random_state=42)
        
    # grid parameters
    #param_grid = dict( n_estimators = [10], max_depth =[10,40,None] )
    
    #score
    #scorer = metrics.make_scorer( metrics.r2_score, greater_is_better = True) 
    
    # fit
    #estimator = GridSearchCV( RFR(random_state = 42,  n_jobs = -1), param_grid, verbose = 0, cv = 3, n_jobs= 1, scoring=scorer )
    #estimator.fit(X_train, y_train)

    
    REG = RFR(random_state = 42,  n_jobs = -1, n_estimators=500, max_depth=None).fit(X_train, y_train)
    
    #REG = estimator.best_estimator_
        
        
    train_e = np.round( 100 * metrics.mean_absolute_percentage_error(y_train, REG.predict(X_train)),2 )
    
    test_e  = np.round( 100 * metrics.mean_absolute_percentage_error(y_test, REG.predict(X_test)),2 )
    return train_e, test_e

def matrix(FEATURES):
    powers = acidoCEST_ML['SatPower(uT)'].unique()
    times = acidoCEST_ML['SatTime(ms)'].unique()
    
    train_results = []
    test_results = []
    
    for p in powers:
        for s in times:
            filt = (acidoCEST_ML['SatPower(uT)'] == p) & (acidoCEST_ML['SatTime(ms)']  == s)
            e1, e2 = train_RFR_without_split(acidoCEST_ML[FEATURES][filt], acidoCEST_ML['pH'][filt])
            train_results.append((p,s,e1))
            test_results.append((p,s,e2))
    
    M_train = pd.DataFrame(train_results,columns=['Sat Power (uT)','Sat Time (ms)','Error']).pivot(index='Sat Power (uT)', columns='Sat Time (ms)', values='Error') 
    M_test = pd.DataFrame(test_results,columns=['Sat Power (uT)','Sat Time (ms)','Error']).pivot(index='Sat Power (uT)', columns='Sat Time (ms)', values='Error')
    return M_train, M_test
    


# ### 4.2 and 5.6

# In[ ]:


M1_train, M1_test  = matrix(FEATURES = ['4.2','5.6'])


# In[ ]:


_ , axs = plt.subplots(dpi=100,ncols=2, figsize=(16,8) )
sns.heatmap( M1_train.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True, vmin= 2, vmax= 5.0,ax =axs[0], cmap='jet')
sns.heatmap( M1_test.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True, vmin= 2, vmax= 5.0,ax =axs[1], cmap='jet')
#plt.title('Mean absolute error (%) for regression of pH \n 4.2 and 5.6 ppm ')


# ### All CEST

# In[ ]:


M2_train, M2_test = matrix(FEATURES = acidoCEST_ML.columns[7::] )


# In[ ]:


_ , axs = plt.subplots(dpi=100,ncols=2, figsize=(16,8) )
sns.heatmap( M2_train.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True, vmin= 2, vmax= 5.0,ax =axs[0], cmap='jet')
sns.heatmap( M2_test.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True, vmin= 2, vmax= 5.0,ax =axs[1], cmap='jet')
#plt.title('Mean absolute error (%) for regression of pH \n 4.2 and 5.6 ppm ')


# ### All data

# In[ ]:


all_features = list(acidoCEST_ML.drop(['pH'],axis=1).columns)
M3_train, M3_test = matrix(FEATURES = all_features )


# In[ ]:


_ , axs = plt.subplots(dpi=100,ncols=2, figsize=(16,8) )
sns.heatmap( M3_train.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True, vmin= 2, vmax= 5.0,ax =axs[0], cmap='jet')
sns.heatmap( M3_test.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True, vmin= 2, vmax= 5.0,ax =axs[1], cmap='jet')
# plt.title('Mean absolute error (%) for regression of pH \n All features')


# ### Applying model outside the ranges of Sat power and Sat time used for training

# In[ ]:


cest_cols = list(acidoCEST_ML.columns[7::])
power = 3.0
time =  3000

filt = (acidoCEST_ML['SatPower(uT)'] == 3.0) & (acidoCEST_ML['SatTime(ms)'] == 3000)
REG = RFR(random_state = 42,  n_jobs = -1, n_estimators=500, max_depth=None).fit(acidoCEST_ML[filt][cest_cols], acidoCEST_ML.pH[filt])

powers = acidoCEST_ML['SatPower(uT)'].unique()
times = acidoCEST_ML['SatTime(ms)'].unique()

Q = []
for p in powers:
    for s in times:
        filt   = (acidoCEST_ML['SatPower(uT)'] == p) & (acidoCEST_ML['SatTime(ms)']  == s)
        y_hat  = REG.predict(acidoCEST_ML[filt][cest_cols])
        y_true = acidoCEST_ML[filt]['pH']
        
        e = np.round( metrics.r2_score(y_true, y_hat),2 )
        Q.append((p,s,e))

MQ = pd.DataFrame(Q,columns=['Sat Power (uT)','Sat Time (ms)','r2_score']).pivot(index='Sat Power (uT)', columns='Sat Time (ms)', values='r2_score')   


# In[ ]:


_, ax = plt.subplots(dpi=100)
sns.heatmap( MQ.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True,  cmap='Set1',ax=ax, vmin=-1 , vmax=1)
plt.title('R2 score for pH regression \n \n The model was trained with all data at 3.0 sec and 3.0 uT \n')


# ### Training model and a small sample of all data -- > Applying model to rest of data

# In[ ]:


cest_cols = list(acidoCEST_ML.columns[7::])
train, test = train_test_split(acidoCEST_ML, test_size=.70)

REG = RFR(random_state = 42,  n_jobs = -1, n_estimators=500, max_depth=None).fit(train[cest_cols], train.pH)

powers = acidoCEST_ML['SatPower(uT)'].unique()
times = acidoCEST_ML['SatTime(ms)'].unique()

Q2 = []
for p in powers:
    for s in times:
        filt   = (test['SatPower(uT)'] == p) & (test['SatTime(ms)']  == s)
        y_hat  = REG.predict(test[filt][cest_cols])
        y_true = test[filt]['pH']
        
        e = np.round( metrics.r2_score(y_true, y_hat),2 )
        Q2.append((p,s,e))

MQ2 = pd.DataFrame(Q2,columns=['Sat Power (uT)','Sat Time (ms)','r2_score']).pivot(index='Sat Power (uT)', columns='Sat Time (ms)', values='r2_score')   


# In[ ]:


_, ax = plt.subplots(dpi=100)
sns.heatmap( MQ2.sort_values('Sat Power (uT)',ascending=False) ,linewidths=1, annot=True,  cmap='Set1',ax=ax, vmin=-1 , vmax=1)
plt.title('R2 score for pH regression \n \n The model was trained with 30% of all data \n')

