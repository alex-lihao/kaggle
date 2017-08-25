import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from datetime import datetime
from sklearn.grid_search import GridSearchCV
print('Loading data ...')
sample = pd.read_csv('sample_submission.csv')
prop = pd.read_csv('properties_2016.csv',low_memory=False)
train = pd.read_csv('train_2016_v2.csv')
print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)
c=0

for value in train['transactiondate']:
    ss=datetime.strptime(value,'%Y-%m-%d')
    ss=ss.year*100+ss.month
    train.set_value(c,'transactiondate',ss)
    c+=1

print('Creating training set ...')
df_train = train.merge(prop, how='left', on='parcelid')
df_train['taxdelinquencyyear'].fillna('0',inplace=True)
df_train['buildingqualitytypeid'].fillna(df_train['buildingqualitytypeid'].mean,inplace=True)
df_train['calculatedbathnbr'].fillna(df_train['bathroomcnt'],inplace=True)
df_train['fireplacecnt'].fillna('0',inplace=True)
df_train['garagecarcnt'].fillna('0',inplace=True)
df_train['garagetotalsqft'].fillna('0',inplace=True)
df_train['hashottuborspa'].fillna('FALSE',inplace=True)
df_train['hashottuborspa'].fillna('13',inplace=True)
df_train['poolcnt'].fillna('0',inplace=True)
df_train['unitcnt'].fillna('1',inplace=True)
df_train['numberofstories'].fillna('1',inplace=True)
df_train=pd.get_dummies(df_train,columns=['regionidcounty','regionidcity'])
df_train['yearbuilt'].fillna(df_train['yearbuilt'].mean,inplace=True)
df_train=df_train.drop(['yardbuildingsqft17','poolsizesum','threequarterbathnbr','propertyzoningdesc','regionidneighborhood','taxdelinquencyflag','basementsqft','architecturalstyletypeid','finishedfloor1squarefeet','finishedsquarefeet15','finishedsquarefeet50','fireplaceflag','storytypeid','typeconstructiontypeid','finishedsquarefeet6','decktypeid','finishedsquarefeet13','buildingclasstypeid','yardbuildingsqft26'],axis=1)
dx_train =df_train.drop(['logerror'],axis=1)
x_train=dx_train
dy_train = df_train['logerror'].values
y_train=dy_train
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = dx_train[:split], y_train[:split], dx_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
cv_train=xgb.DMatrix(dx_train, label=dy_train)
del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.1
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 8
params['silent'] = 1
params['min_child_weight']=1
params['subsample']=0.8
params['colsample_bytree']=0.8
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5], 'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'reg:linear'}
optimized_GBM = GridSearchCV(xgb.XGBRegressor(), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = 1) 
optimized_GBM.fit(dx_train, dy_train)