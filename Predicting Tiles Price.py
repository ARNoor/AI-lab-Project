#Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv('Ceramic Tiles Dataset.csv')

#Checking the number of missing values for each attribute
dataset.isna().sum()

#Getting only those features that has got less or no empty values
dataset_new = dataset.iloc[:, [0,3,4,5,7,9,11]]

#Values of the selected feature
dataset["Size (mm)"]

#Filled nan values with the most frequent value in the feature
dataset_new = dataset_new.fillna(dataset_new.mode().iloc[0])
#Again checking the number of missing values for each attribute
dataset_new.isna().sum()

#To see the unique values in the feature
dataset_new.Material.unique()
#Replacing Porcelain Clay with Porcelain as they mean the same
dataset_new['Material'] = dataset_new['Material'].replace(['Porcelain Clay'],['Porcelain'])

#Encoding to get numerical values
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('Ceramic Tiles Dataset', OneHotEncoder(), [0, 4])], remainder = 'passthrough')
dataset_new = ct.fit_transform(dataset_new)

#Data preprocessing is done

#Generating heatmap to see the correlation among the features
df = pd.DataFrame(dataset_new)
corr = df.corr()
sns.heatmap(corr, annot=True)

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

dt_MAE = []
dt_MSE = []
dt_RMSE = []
dt_r2f = []

rf_MAE = []
rf_MSE = []
rf_RMSE = []
rf_r2f = []

br_MAE = []
br_MSE = []
br_RMSE = []
br_r2f = []

svm_MAE = []
svm_MSE = []
svm_RMSE = []
svm_r2f = []

for train_index, test_index in kf.split(dataset_new):
    dataset_train, dataset_test = dataset_new[train_index], dataset_new[test_index]
    #Feature_scaling
    X_train = dataset_train[:,[1,3,4,5,6,7,8,9,10]]
    y_train = dataset_train[:, [11]]
    X_test= dataset_test[:,[1,3,4,5,6,7,8,9,10]]
    y_test= dataset_test[:, [11]]
    
    #Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #Decision Tree
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    dt_MAE.append(metrics.mean_absolute_error(y_test,y_pred))
    dt_MSE.append(metrics.mean_squared_error(y_test,y_pred))
    dt_RMSE.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    dt_r2f.append(metrics.r2_score(y_test,y_pred))
    
    #Random Forrest
    regressor = RandomForestRegressor(n_estimators=10)
    regressor.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = regressor.predict(X_test)
    rf_MAE.append(metrics.mean_absolute_error(y_test,y_pred))
    rf_MSE.append(metrics.mean_squared_error(y_test,y_pred))
    rf_RMSE.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    rf_r2f.append(metrics.r2_score(y_test,y_pred))
    
    #Bayesian Ridge
    regressor = linear_model.BayesianRidge()
    regressor.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = regressor.predict(X_test)
    br_MAE.append(metrics.mean_absolute_error(y_test,y_pred))
    br_MSE.append(metrics.mean_squared_error(y_test,y_pred))
    br_RMSE.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    br_r2f.append(metrics.r2_score(y_test,y_pred))
    
    #Support Vector Machine
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = regressor.predict(X_test)
    svm_MAE.append(metrics.mean_absolute_error(y_test,y_pred))
    svm_MSE.append(metrics.mean_squared_error(y_test,y_pred))
    svm_RMSE.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    svm_r2f.append(metrics.r2_score(y_test,y_pred))
    
print("Comparison of the performance scores of the models:\n")
print("Model Name                  MAE          MSE          RMSE        R2-score")
print("---------------------------------------------------------------------------")
print("Decision Tree              ",format(sum(dt_MAE)/5,".3f"), "      ",format(sum(dt_MSE)/5,".3f"),"      ",format(sum(dt_RMSE)/5,".3f"),"      ",format(sum(dt_r2f)/5,".3f"))
print("Random Forest              ",format(sum(rf_MAE)/5,".3f"), "      ",format(sum(rf_MSE)/5,".3f"),"      ",format(sum(rf_RMSE)/5,".3f"),"      ",format(sum(rf_r2f)/5,".3f"))
print("Bayesian Ridge             ",format(sum(br_MAE)/5,".3f"), "      ",format(sum(br_MSE)/5,".3f"),"      ",format(sum(br_RMSE)/5,".3f"),"      ",format(sum(br_r2f)/5,".3f"))
print("Support Vector Regression  ",format(sum(svm_MAE)/5,".3f"), "      ",format(sum(svm_MSE)/5,".3f"),"      ",format(sum(svm_RMSE)/5,".3f"),"      ",format(sum(svm_r2f)/5,".3f"))
