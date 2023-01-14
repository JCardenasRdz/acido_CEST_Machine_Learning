#!/usr/bin/env python
# coding: utf-8

# # Regression of pH range using machine learning

# In[1]:


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


# ## Data

# In[2]:


# load data
acidoCEST_ML = pd.read_parquet('../clean_data/acido_CEST_MRI_Iopamidol.parquet.gzip')

# drop columns we cannot measure or that are redundant
cols_2_drop = ['ApproT1(sec)','Temp','ExpB0(ppm)','FILE','Conc(mM)']
acidoCEST_ML = acidoCEST_ML.drop(cols_2_drop, axis = 1)


# In[3]:


# define experimental columns -- > things we measure other than CEST
exper_cols = [ 'ExpB1(percent)', 'ExpB0(Hz)', 'SatPower(uT)',  'SatTime(ms)','ExpT1(ms)', 'ExpT2(ms)']


# In[4]:


# used during development
# acidoCEST_ML  = acidoCEST_ML.sample( n = 1_00)


# In[5]:


acidoCEST_ML.shape


# ## Functions

# In[6]:


# Regression
from sklearn.metrics import confusion_matrix, classification_report
    
def train_lasso(Xdata, pH_observed, create_fig = 1, verbose=0):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(Xdata, pH_observed, test_size=0.30, random_state=42)
    
    # LASSO
    lasso = linear_model.Lasso( max_iter = 5000, random_state=42)

    #pipeline
    pipe = Pipeline(steps=[('Scaler', pp.StandardScaler()), ('lasso', lasso)])

    param_grid ={
            'lasso__alpha': np.linspace(.01,1,20),
            'lasso__fit_intercept': [True,False]
            }

    scorer = metrics.make_scorer( metrics.mean_absolute_percentage_error, greater_is_better=False) 


    estimator = GridSearchCV(pipe, param_grid, verbose = verbose, cv = 3, n_jobs= 6, scoring=scorer )


    # fit
    estimator.fit(X_train, y_train)

    score_train = np.round( 100 * metrics.mean_absolute_percentage_error(y_train , estimator.predict(X_train)), 2)
    score_test  = np.round( 100 * metrics.mean_absolute_percentage_error(y_test , estimator.predict(X_test)), 2)
    
    #score on pH units
    score_pH_train = np.mean( np.abs(  y_train -  estimator.predict(X_train) ) )
    score_pH_test = np.mean( np.abs(  y_test -  estimator.predict(X_test) ) )

    m  = f'\n mean absolute percentage error: \n train = {score_train} % \n test  = {score_test} %'
    m2 = f'\n pH units |error : \ntrain = {score_pH_train:.3f} \n test = {score_pH_test:.3f}'

    if create_fig == 1:
        print('------------------ LASSO ------------------ ')
        print()
        #print(', '.join(Xdata.columns.to_list()))
        print(m,'\n',m2,'\n')

        plt.figure()
        sns.regplot(x = estimator.best_estimator_.predict(X_train), y =  y_train)
        plt.xlabel('\n Predicted pH')
        plt.ylabel('Measured pH \n')
        plt.title('Train LASSO')

        plt.figure()
        sns.regplot(x = estimator.best_estimator_.predict(X_test), y =  y_test)
        plt.xlabel('\n Predicted pH')
        plt.ylabel('Measured pH \n')
        plt.title('Test LASSO')
    
    
    scores = dict()
    scores['mae_train'] = score_train
    scores['mae_test'] = score_test
    scores['pH_train'] = score_pH_train
    scores['pH_test'] = score_pH_test
    
    test_data = dict()
    test_data['X'] = X_test
    test_data['Y'] = y_test
    
    return estimator.best_estimator_, scores, test_data


# ## LASSO

# ### -- 4.2 and 5.6 only

# In[7]:


get_ipython().run_cell_magic('time', '', "\nX = acidoCEST_ML[['4.2','5.6']]\nY = acidoCEST_ML.pH\nreg_lr, _ , _ = train_lasso(X,Y, verbose=1)\n")


# ### -- All Saturation Frequencies

# In[8]:


get_ipython().run_cell_magic('time', '', "\nXdata = acidoCEST_ML.drop(exper_cols + ['pH'],axis=1)\nYdata = acidoCEST_ML.pH \nprint(Xdata.shape)\nreg_lr2, _ , _ = train_lasso(Xdata,Ydata, verbose=1)\n")


# ### -- All data

# In[9]:


get_ipython().run_cell_magic('time', '', "\nXdata = acidoCEST_ML.drop( ['pH'],axis=1) \nYdata = acidoCEST_ML.pH\nprint(Xdata.shape)\nlasso_all_data, _, _  = train_lasso(Xdata,Ydata, verbose=1)\n\nlasso_all_data['lasso']\n")


# ### -- selected features

# #### | - all selected

# In[10]:


get_ipython().run_cell_magic('time', '', "\nreg_lasso = lasso_all_data['lasso']\n\nC = pd.DataFrame( reg_lasso.coef_ , columns=['Lasso Coeff'], index = Xdata.columns)\n_, ax = plt.subplots(dpi = 100)\n\nC[reg_lasso.coef_ != 0].plot(kind='bar', ax = ax)\nplt.title('Non-zero Lasso Coefficients \\n')\nplt.xlabel('\\n Feature')\nplt.ylabel('Value \\n')\n")


# In[11]:


get_ipython().run_cell_magic('time', '', "\nselected_cols_lasso = list(C[reg_lasso.coef_ != 0].index) # all data are non zero\n\nXdata = acidoCEST_ML[selected_cols_lasso]\nYdata = acidoCEST_ML.pH\n\nlasso_selected, scores, tes_data_selected_features = train_lasso(Xdata[selected_cols_lasso],Ydata, verbose=1)\n\nlasso_selected['lasso']\n")


# ## Features selected

# In[12]:


pd.Series(selected_cols_lasso)


# #### | - sat power vs  sat time

# In[13]:


get_ipython().run_cell_magic('time', '', "\nsat_times = acidoCEST_ML['SatTime(ms)'].unique()\n#sat_times.sort()\n\nsat_powers = acidoCEST_ML['SatPower(uT)'].unique()\n#sat_powers.sort()\n\nM_lasso = pd.DataFrame(np.zeros( (len(sat_powers), len(sat_powers))), index=sat_powers, columns=sat_times)\n\nfor t in sat_times:\n    for p in sat_powers:\n        D = acidoCEST_ML[(acidoCEST_ML['SatTime(ms)'] == t) & (acidoCEST_ML['SatPower(uT)'] == p) ]\n\n        Xdata = D[selected_cols_lasso]\n        Ydata = D['pH']\n        \n        lasso_selected, scores, _ = train_lasso(Xdata,Ydata, create_fig=0)\n        M_lasso.loc[p,:][t] = scores['mae_test']\n\nM_lasso.columns = [str(int(x/1000)) + ' sec'  for x in M_lasso.columns]\nM_lasso.index   = [f'{int(x)} uT' for  x in M_lasso.index ]\n\n_, ax = plt.subplots(dpi=200)\n\ncols_ = list(M_lasso.columns)\ncols_.sort()\n\nidx = list(M_lasso.index)\nidx.sort()\n\n\nsns.heatmap(M_lasso.loc[idx,:][cols_], cmap='rainbow',annot=True,linewidths=.1,ax=ax)\nplt.xlabel('\\n Saturation Time')\nplt.ylabel('Saturation Power \\n')\nplt.title('Mean Abs. Error (%) in the pH estimate \\n LASSO \\n')\n\nplt.savefig(f'../figs/MATRIX_regression_LASSO.png',bbox_inches='tight')\n")


# ## Predictions for test data using selected features

# In[14]:


yhat = lasso_selected.predict( tes_data_selected_features['X'] )

out = pd.DataFrame(yhat, columns=['Predicted_pH_LASSO_TEST'])
out['Measured_pH_TEST'] = tes_data_selected_features['Y'].values
out.to_csv('../outputs/lasso_reg_TEST_pred.csv')

out

