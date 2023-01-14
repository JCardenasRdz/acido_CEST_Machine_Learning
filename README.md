# Measuring pH using Machine Learning and Chemical Exchange Saturation Transfer (CEST) MRI

 Code to reproduce CAMEL's paper on how to measure pH using Machine Learning (ML) and acidoCEST-MRI
 
# Organization
All the code to clean the data and train the model are under the `./notebooks`.
We used two machine learning methods (ML) for each task/goal:

- Classification 
    - outcomes predicted (separate models): pH > 6.5, pH > 7.0
    - Logistic Regression (with L1 penalty)
    - Random Forest Classifier
    

- Regression 
    - Linear Regression (with L1 penalty) = LASSO
    - Random Forest Regressort
    

 We also created the model using fourt different set of features:
 
 - CEST signal ([Mo-Ms]/Ms)at suration offsets of 4.2 and 5.6 ppm
 - All saturation offsets (81 total)
 - All data : B1, B0, T1, T2, Sat Power, Sat time, and  CEST signal at 81 sat offsets
 - Selected features by: LASSO (non-zero coefficients), Feature importance (top 20)
 
 
 
## Computation wall times (seconds)

- Classfication (total time for both pH cut offs)

| Condition        	| L1 Log Reg 	| Random Forest Classifier 	
|------------------	|------------	|--------------------------	|
| 4.2 and 5.6 pppm 	| 3.22       	| 15.8                    	| 
| All sat. freqs   	| 655        	| 121                      	|
| All data         	| 715        	| 123                     	| 
| selected features | 48.5        	| 57.4                     	| 


- Regression

| Condition        	|   LASSO 	    | Random Forest Classifier 	
|------------------	|------------	|--------------------------	|
| 4.2 and 5.6 pppm 	| 2.21       	| 35.6                    	| 
| All sat. freqs   	| 3.63        	| 616                      	|
| All data         	| 4.35        	| 601                      	| 
| selected features | 1.53        	| 144                    	| 



## Installation
 
Fist you need to set your conda environment up

``` bash
(base) %: conda create -n acidoCEST_ML python=3.10.0
(base) %: conda activate acidoCEST_ML
(acidoCEST_ML) %: python -m pip install --upgrade pip
(acidoCEST_ML) %: python -m pip install -r requirements.txt
```
After that you add your conda environment to the kernels for your jupyter notebook (if needed)

```bash
(acidoCEST_ML) %: conda install -c anaconda ipykernel
(acidoCEST_ML) %: python -m ipykernel install --user --name=acidoCEST_ML
```