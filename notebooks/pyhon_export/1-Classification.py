#!/usr/bin/env python
# coding: utf-8

# # Classification of pH range using machine learning

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


# # Load data

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


s = [x for x in acidoCEST_ML.shape]
print( f'data size | rows = {s[0]:,}, cols = {s[1]:,}' )


# In[5]:


# used during development
#acidoCEST_ML  = acidoCEST_ML.sample(frac=.10)


# # Logistic Regression

# In[6]:


# Logistic Regression
from sklearn.metrics import confusion_matrix, classification_report

def classification_metrics(Yexpected, Yhat):
    cm = confusion_matrix(Yexpected, Yhat)
    TN, FP, FN, TP = list(cm.flatten())
    
    metrics = dict (  PPV =  TP / (TP + FP) 
                 , NPV = TN / (TN + FN)
                 , SEN = TP / (TP + FN)
                 , SPC = TN / (TN + FP)
               )
    for m in metrics.keys():
        metrics[m] = np.round(metrics[m],3)
    
    return metrics

def train_logistic_reg(Xdata, pH_observed, pH_cut_off = 7.0, n_cs = 20, maxC = 1):
    
    # cut off > pH
    y = 1*(pH_observed > pH_cut_off)
    # X data
    X = Xdata.copy()
        
    # Logistic
    logistic = linear_model.LogisticRegression(solver='saga', penalty='l1', max_iter=10_000,random_state=42, n_jobs=-1)

    #pipeline
    pipe = Pipeline(steps=[('Scaler', pp.StandardScaler()), ('logistic', logistic)])


    # Training parameters
    Cs = np.logspace(-3, maxC, n_cs)
    
    param_grid ={
    'logistic__C': Cs,
    'logistic__class_weight': ['balanced',None]
    }
    
    
    estimator = GridSearchCV(pipe, param_grid, verbose = 3, cv = 3, n_jobs= 2, scoring=metrics.make_scorer( metrics.balanced_accuracy_score))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Grid Search of Model
    estimator.fit(X_train, y_train)
    
    # CLASSIFC
    classification_per = classification_metrics(y_test, estimator.predict(X_test))
    print('Classification Performance \n')
    [print(f'{key} : {value}') for key, value in classification_per.items()]
    print('---'*10)
    #AUC
    y_probas = estimator.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro=False,    plot_macro=False)
    
    #skplt.metrics.plot_confusion_matrix (y_test, estimator.predict(X_test))
    
    plt.show()
    
    #print( metrics.classification_report(y_test, estimator.predict(X_test)) )
    
    

    return estimator.best_estimator_, X_train, X_test, y_train, y_test

def classify_pH_Log_Reg(X,Y, cut_offs=[6.5,7.0], Ncs= 10):
    # Data Size
    data_size(X)
    
    ## Logistic REgression: Only Zspectra pH = 7.0
    classifiers =dict()
    
    for pH in cut_offs: 
        print('======  pH = {} ====== '.format(pH))
        clf, X_train, X_test, y_train, y_test = train_logistic_reg(X,Y, pH_cut_off = pH, n_cs = Ncs);
        
        classifiers[str(pH)] = clf

    return classifiers

def data_size(DataFrame):
    r, c = DataFrame.shape
    print(f' row = {r:,} | cols = {c:,}')


# ## Number of Cs for Log. Reg

# In[7]:


Ncs = 10


# ## - 4.2 and 5.6 only

# In[8]:


get_ipython().run_cell_magic('time', '', "Xdata_iso = acidoCEST_ML[['4.2','5.6']]\nYdata = acidoCEST_ML.pH\n\ntwo_freqs = classify_pH_Log_Reg(Xdata_iso,Ydata, cut_offs=[6.5,7.0], Ncs=Ncs)\n")


# ## - 4.2 ppm, 5.6 ppm, exp data

# In[9]:


len(['4.2','5.6'] + exper_cols)


# In[10]:


print(exper_cols)


# In[11]:


get_ipython().run_cell_magic('time', '', "\nc = ['4.2','5.6'] + exper_cols\n\nXdata_iso_and_exp = acidoCEST_ML[c]\nYdata = acidoCEST_ML['pH'].apply(lambda x: np.round(x,1))\n\ntwo_freqs_and_exp = classify_pH_Log_Reg(Xdata_iso_and_exp,Ydata, cut_offs=[6.5,7.0], Ncs=Ncs)\n")


# ## - Only Zspectra

# In[12]:


get_ipython().run_cell_magic('time', '', "\nXdata = acidoCEST_ML.drop(exper_cols + ['pH'],axis=1)\nYdata = acidoCEST_ML.pH \n\nonly_zspectra = classify_pH_Log_Reg(Xdata,Ydata, cut_offs=[6.5,7.0], Ncs = Ncs)\n\nprint( f'{len(Xdata.columns)} frequencies' )\nprint( f'{len(exper_cols)} exper cols' )\n")


# ## -  Zspectra  + exp data

# In[13]:


get_ipython().run_cell_magic('time', '', "\nXdata = acidoCEST_ML.drop( ['pH'],axis=1) \nYdata = acidoCEST_ML.pH\n\nall_data = classify_pH_Log_Reg(Xdata,Ydata, cut_offs=[6.5,7.0], Ncs= Ncs)\nprint( len( list(Xdata.columns) ) )\n")


# ## -- Coeff

# In[14]:


plt.style.use('tableau-colorblind10')


# In[15]:


coef = all_data['7.0']['logistic'].coef_

C = pd.DataFrame(1-np.exp(coef.reshape(-1,1)),index=acidoCEST_ML.drop( ['pH'],axis=1).columns, columns=['Coeff' ])
print(np.exp(C.head(7)))
_, ax = plt.subplots(dpi=100, figsize=(16,8))
C.plot(kind='bar',ax=ax,rot=90)
plt.title('Logistic Regresson Coefficients for pH cut off = 7.0 \n')
plt.savefig('../figs/Figure_01_Classification.png',bbox_inches='tight')
plt.show()


# In[16]:


coef = all_data['6.5']['logistic'].coef_

C = pd.DataFrame(1-np.exp(coef.reshape(-1,1)),index=acidoCEST_ML.drop( ['pH'],axis=1).columns, columns=['Coeff' ])
print(np.exp(C.head(7)))
_, ax = plt.subplots(dpi=100, figsize=(16,8))
C.plot(kind='bar',ax=ax,rot=90)
plt.title('Logistic Regresson Coefficients for pH cut off = 6.5 \n')

plt.savefig('../figs/Figure_02_Classification.png',bbox_inches='tight')
plt.show()


# ## -- Feature Selection and Log Reg on selected features

# In[17]:


get_ipython().run_cell_magic('time', '', "\ndef select_cols(coefficients):\n    d = np.exp(coefficients)\n    f= (d >1.90) | (d < 0.10)\n    cols = np.array(Xdata.columns.values)\n    new_cols = cols[f.reshape(-1,)]\n\n    return new_cols\n    \ncoef1 = all_data['7.0']['logistic'].coef_\ncoef2 = all_data['6.5']['logistic'].coef_\n\ns1 = set(select_cols(coef1))\ns2 = select_cols(coef2)\n\nnew_cols = list(s1.union(s2))\nnew_cols.sort()\n\nprint(len(new_cols))\n\nXdata_small = acidoCEST_ML.drop( ['pH'],axis=1)[new_cols]\nYdata = acidoCEST_ML.pH\n\nall_data_small = classify_pH_Log_Reg(Xdata_small,Ydata, cut_offs=[6.5,7.0], Ncs=Ncs)\n")


# In[18]:


print(f'The following features were selected for Logistic Regression: \n {new_cols}')


# In[19]:


# save output
all_data_small_change_in_odds = pd.DataFrame(all_data_small['6.5']['logistic'].coef_.reshape(-1,1),index=new_cols, columns=['6.5'])
all_data_small_change_in_odds['7.0'] = all_data_small['7.0']['logistic'].coef_.reshape(-1,1)
all_data_small_change_in_odds = all_data_small_change_in_odds.applymap(lambda f: np.exp(f) - 1).round(2)
all_data_small_change_in_odds.to_csv('../outputs/classification_change_in_odds_selected.csv')


# In[20]:


#  save pipelines
from joblib import dump, load
dump(all_data_small['6.5'], '../models/pH_classication_pipeline_6p5.joblib') 
dump(all_data_small['7.0'], '../models/pH_classication_pipeline_7p0.joblib') 


# # Random Forest

# In[21]:


from sklearn.ensemble import RandomForestClassifier as RFC

def train_RFC(Spectra, pH_observed, pH_cut_off = 7.0):
    # cut off > pH
    y = 1*(pH_observed > pH_cut_off)
    # X data
    X = Spectra.copy()
        
    # Logistic
    rf = RFC(random_state=42, n_jobs=10)
    
    param_grid ={
    'n_estimators': [5,10,20,50,100],
    'max_depth': [2,4,8,10,20] ,
        'class_weight':['balanced',None] ,
        'criterion':['entropy','gini']
    }
    
    estimator = GridSearchCV(rf, param_grid, verbose = 1, cv = 3, n_jobs=6, scoring=metrics.make_scorer( metrics.balanced_accuracy_score))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Grid Search of Model
    estimator.fit(X_train, y_train)
    
    
    # CLASSIFC
    classification_per = classification_metrics(y_test, estimator.predict(X_test))
    print('Classification Performance \n')
    [print(f'{key} : {value}') for key, value in classification_per.items()]
    print('---'*10)
    
    #AUC
    y_probas = estimator.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas)

    plt.show()
    
    return estimator.best_estimator_
    
    


# ## -- 4.2 and 5.6 only

# In[22]:


get_ipython().run_cell_magic('time', '', "\npH = 6.5\n\nprint('==='*20)\nprint('############ pH = {}############'.format(pH))\nXdata = acidoCEST_ML[['4.2','5.6']]\nYdata = acidoCEST_ML.pH\nRF_two_freqs_65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\npd.DataFrame(RF_two_freqs_65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar')\n")


# In[23]:


get_ipython().run_cell_magic('time', '', "\npH = 7.0\n\nprint('==='*20)\nprint('############ pH = {}############'.format(pH))\nXdata = acidoCEST_ML[['4.2','5.6']]\nYdata = acidoCEST_ML.pH\nRF_two_freqs_70 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\npd.DataFrame(RF_two_freqs_70.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar')\n")


# ## -- 4.2 and 5.6  and  non-CEST exp data

# In[24]:


get_ipython().run_cell_magic('time', '', "\nc = ['4.2','5.6'] + exper_cols\npH = 6.5\n\nXdata = acidoCEST_ML[c]\nYdata = acidoCEST_ML.pH\n\nRF_two_freqs_and_exp65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\npd.DataFrame(RF_two_freqs_and_exp65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar')\n")


# In[25]:


print(RF_two_freqs_and_exp65)


# In[26]:


get_ipython().run_cell_magic('time', '', "\nc = ['4.2','5.6'] + exper_cols\npH = 7.0\n\nXdata = acidoCEST_ML[c]\nYdata = acidoCEST_ML.pH\n\nRF_two_freqs_and_exp70 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\npd.DataFrame(RF_two_freqs_and_exp70.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar')\n")


# ## -- All sat freqs 

# In[27]:


get_ipython().run_cell_magic('time', '', "\npH = 6.5\n\nXdata = acidoCEST_ML.drop(exper_cols + ['pH'],axis=1)\nYdata = acidoCEST_ML.pH \n\nRF_all_freqs65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_all_freqs65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# In[28]:


get_ipython().run_cell_magic('time', '', "\npH = 7.0\n\nXdata = acidoCEST_ML.drop(exper_cols + ['pH'],axis=1)\nYdata = acidoCEST_ML.pH \n\nRF_all_freqs70 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_all_freqs70.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# ## -- All data

# In[29]:


get_ipython().run_cell_magic('time', '', "\n\npH = 6.5\n\nXdata = acidoCEST_ML.drop(['pH'],axis=1)\nYdata = acidoCEST_ML.pH \n\nRF_all_data65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_all_data65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# In[30]:


get_ipython().run_cell_magic('time', '', "\npH = 7.0\n\nXdata = acidoCEST_ML.drop(['pH'],axis=1)\nYdata = acidoCEST_ML.pH \n\nRF_all_data70 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_all_data70.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# ## -- Selected features

# In[31]:


s1 = set(pd.DataFrame(RF_all_data65.feature_importances_, index=Xdata.columns, columns=['Imp']).sort_values('Imp',ascending=False).head(20).index)
s2 = set(pd.DataFrame(RF_all_data70.feature_importances_, index=Xdata.columns, columns=['Imp']).sort_values('Imp',ascending=False).head(20).index)
sel_f = list(s1.union(s1))
sel_f.sort()
print(f'The following features were selected for Random Forest are : \n {sel_f}')


# In[32]:


len(sel_f)


# In[33]:


get_ipython().run_cell_magic('time', '', "\npH = 6.5\n\nXdata = acidoCEST_ML[sel_f]\nYdata = acidoCEST_ML.pH \n\nRF_set_features_65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_set_features_65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# In[34]:


get_ipython().run_cell_magic('time', '', "\npH = 7.0\n\nXdata = acidoCEST_ML[sel_f]\nYdata = acidoCEST_ML.pH \n\nRF_set_features_70 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_set_features_70.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# In[35]:


# save models for RF
# clf = load('filename.joblib') 
# https://scikit-learn.org/stable/modules/model_persistence.html
dump( RF_set_features_65, '../models/pH_classication_random_forest_6p5.joblib') 
dump( RF_set_features_70, '../models/pH_classication_random_forest_7p0.joblib') 


# ## Tree Vis

# In[36]:


get_ipython().run_cell_magic('capture', '', "import matplotlib.pyplot as plt\nfrom sklearn.tree import plot_tree\n\n\nnew_names = []\n\nfor i,x in enumerate(sel_f):\n    if x.split('.')[0].isnumeric() == True:\n        new_names.append( f'{x} ppm' )\n        \n    else:\n        new_names.append( x.split('(')[0].split('Exp')[1] )\n\n\n        \n        \ntree=RF_set_features_70.estimators_[9]\n\nclasses = list( pd.Series(tree.classes_).replace([0,1],[' pH < 7', 'pH > 7']).values )\n\nfig = plt.figure(dpi=500)\nplot_tree( tree,  feature_names = new_names, class_names= classes,\n          filled=True, impurity=False,  max_depth = 4, fontsize = 4, label='none', proportion =True,\n          rounded=False);\n\nplt.savefig('../figs/Tree_example_max_depth_04_tree09.png',bbox_inches='tight')\n")


# In[37]:


from  sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(tree, 
                 out_file='tree09_04_levels.dot', 
                feature_names = new_names,
                class_names = classes,
                max_depth = 4 ,
                rounded = True, proportion = False, 
                precision = 2, filled = True, leaves_parallel=True, rotate=True)


# In[38]:


from sklearn.tree import export_text

tree_as_text_09 = export_text(RF_set_features_70.estimators_[9]
                  ,  max_depth = 4
                  ,  decimals = 3
                  ,  spacing = 3
                  , feature_names = new_names 
                  , show_weights=True )   

print(tree_as_text_09)


# ## -- Selected features (without B1)

# ### - LASSO
# 
# not needed because B1 was not selected

# ### - random forest

# In[39]:


get_ipython().run_cell_magic('time', '', "\npH = 6.5\npreds = [x for x in sel_f  if 'B1' not in x ]\n\nXdata = acidoCEST_ML[preds]\nYdata = acidoCEST_ML.pH \n\nRF_set_features_65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_set_features_65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# In[40]:


get_ipython().run_cell_magic('time', '', "\npH = 7.\npreds = [x for x in sel_f  if 'B1' not in x ]\n\nXdata = acidoCEST_ML[preds]\nYdata = acidoCEST_ML.pH \n\nRF_set_features_65 = train_RFC(Xdata,Ydata, pH_cut_off= pH)\n\n_ , ax = plt.subplots(dpi=100, figsize=(14,8))\npd.DataFrame(RF_set_features_65.feature_importances_, index=Xdata.columns, columns=['Imp']).plot(kind='bar',ax=ax)\n")


# In[ ]:




