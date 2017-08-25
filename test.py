import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from datetime import datetime
print('Loading data ...')
sample = pd.read_csv('sample_submission.csv')
prop = pd.read_csv('properties_2016.csv')
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
y_train = df_train['logerror'].values
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = dx_train[:split], y_train[:split], dx_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 6
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
sample_1=pd.DataFrame(columns=['parcelid','transactiondate'])
for t in sample.columns[sample.columns !='parcelid']:
    for p in sample['parcelid']:
        sample_1.append([p,t])
df_test = sample_1.merge(dx_train, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test,sample_1, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f')