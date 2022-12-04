import gc
import io
import numpy as np
import pandas as pd

import sklearn
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
import sklearn.ensemble as es
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns

data=pd.read_csv('../train_dataset/train_public.csv')
test1=pd.read_csv('../test_public.csv')
pd.set_option('display.max_columns',None)
print(data.columns)
#查看数据状态
print(data.head())
#查看数据类型
print(data.info())
print(data.groupby(['isDefault']).size().reset_index())
#输出每行的空缺值数目
for col in data.columns:
    print('%s: %d' % (col,sum(data[col].isnull()==True)))
list_nan=['work_year','f0','f1','f2','f3','f4']
for col in list_nan:
    print('%s: %f' % (col,sum(data[col].isnull())/data.shape[0]))
#采用众数填充
data1=data.fillna(data.mean()).copy()
data1['work_year']=data['work_year'].fillna(data['work_year'].mode().iloc[0]).copy()
test1.fillna(test1.mode().iloc[0],inplace=True)
#采用分类、聚类的方式填充，有空写

#删除空缺值，暂时先用这个
#data1=data.dropna().copy()
#输出重复的数据总数
print('Numbers of Duplicated:',data['loan_id'].duplicated().sum())
#将class,employer_type,industry编码为数值数据，采用序号编码
skpe=skp.LabelEncoder()
skpe.fit(data1['class'])
data1['class']=skpe.transform(data1['class'])
test1['class']=skpe.transform(test1['class'])
skpe.fit(data1['employer_type'])
data1['employer_type']=skpe.transform(data1['employer_type'])
test1['employer_type']=skpe.transform(test1['employer_type'])
skpe.fit(data1['industry'])
data1['industry']=skpe.transform(data1['industry'])
test1['industry']=skpe.transform(test1['industry'])
print(data1.head())
#将work_year转为数字时间,0为不足一年，10为超过10年
work_year=[]
for i in data1['work_year'].values.tolist():
    tmp=str(i).split(' ')
    if tmp[0]=='10+':
        work_year.append(10)
    elif tmp[0]=='<':
        work_year.append(0)
    else:
        work_year.append(int(tmp[0]))
work_year=pd.Series(work_year)
data1['work_year']=work_year.values
#测试集
work_year=[]
for i in test1['work_year'].values.tolist():
    tmp=str(i).split(' ')
    if tmp[0]=='10+':
        work_year.append(10)
    elif tmp[0]=='<':
        work_year.append(0)
    else:
        work_year.append(int(tmp[0]))
test1['work_year']=work_year
#将earlies_credit_date，issue_date提取出年月两个数据。
def earlies_credit_year_work(x):
    if x<22:
        x+=2000
    elif 22<x<100:
        x+=1900
    return x
data1['issue_date']=pd.to_datetime(data1['issue_date'],format='%Y/%m/%d')
earlies_credit_month=[]
earlies_credit_year=[]
earlies_credit_month_dict={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
for i in data1['earlies_credit_mon'].values.tolist():
    tmp=str(i).split('-')
    if tmp[0] in earlies_credit_month_dict:
        earlies_credit_month.append(earlies_credit_month_dict[tmp[0]])
        earlies_credit_year.append(earlies_credit_year_work(int(tmp[1])))
    elif tmp[1] in earlies_credit_month_dict:
        earlies_credit_month.append(earlies_credit_month_dict[tmp[1]])
        earlies_credit_year.append(2000)
data1['earlies_credit_mon']=earlies_credit_month
data1['earlies_credit_year']=earlies_credit_year
data1['issue_year']=data1['issue_date'].dt.year
data1['issue_month']=data1['issue_date'].dt.month
print(data1.head())

#测试集
test1['issue_date']=pd.to_datetime(test1['issue_date'],format='%Y/%m/%d')
earlies_credit_month=[]
earlies_credit_year=[]
for i in test1['earlies_credit_mon'].values.tolist():
    tmp=str(i).split('-')
    if tmp[0] in earlies_credit_month_dict:
        earlies_credit_month.append(earlies_credit_month_dict[tmp[0]])
        earlies_credit_year.append(earlies_credit_year_work(int(tmp[1])))
    elif tmp[1] in earlies_credit_month_dict:
        earlies_credit_month.append(earlies_credit_month_dict[tmp[1]])
        earlies_credit_year.append(2022)
test1['earlies_credit_mon']=earlies_credit_month
test1['earlies_credit_year']=earlies_credit_year
test1['issue_year']=test1['issue_date'].dt.year
test1['issue_month']=test1['issue_date'].dt.month
prediction = test1[['loan_id']]
prediction['isDefault'] = 0
#删除policy_code与issue_date
data1=data1.drop('issue_date',axis=1)
data1=data1.drop('policy_code',axis=1)#如果用另一个表删除这句

test1=test1.drop('issue_date',axis=1)
test1=test1.drop('policy_code',axis=1)#如果用另一个表删除这句
# test1=test1.drop('loan_id',axis=1)
# test1=test1.drop('user_id',axis=1)

#采用Z-score查看异常情况
for col in data1.columns:
    Z=(data1[col]-data1[col].mean())/data1[col].std()
    num=Z[(Z < -3) | (Z > 3)].size
    print('%s: %f ' % (col,num))


# for i in range(len(data1.columns.values.tolist())):
#     plt.figure(figsize=(20,20))
#     plt.title(data1.columns[i],fontdict={'weight':'normal','size': 40})
#     plt.boxplot(data1.iloc[:,i],showfliers=True)
#     plt.savefig('../boxplot/{0}.png'.format(i))
#     plt.show()
#early_return_amone_3mon,early_return_amount做个DBSCAN除异常，#删了神经网络效果更差，随机森林看不出问题
# data1.drop(data1[data1['early_return_amount_3mon'].max()==data1['early_return_amount_3mon']].index.values,axis=0)
# data1.drop(data1[data1['early_return_amount'].max()==data1['early_return_amount']].index.values,axis=0)
#删除house_exist最高值,
# data1.drop(data1[data1['house_exist'].max()==data1['house_exist']].index.values)
# data1.drop(data1[data1['total_loan']>38000].index.values)

#看变量相关性并绘制图片，total_loan和monthly_payment，class和interest线性强相关
data1=pd.read_csv('./feature.csv')
# test1=pd.read_csv('./feature_test.csv')
data1=data1.drop(data1[data1['debt_loan_ratio'].max()==data1['debt_loan_ratio']].index.values)
data1=data1.drop(data1[data1['debt_loan_ratio'].max()==data1['debt_loan_ratio']].index.values)
data1=data1.drop(data1[data1['recircle_b']>500000].index.values)
data1=data1.drop(data1[data1['house_exist'].max()==data1['house_exist']].index.values)
data1['earlies_credit_year']=data1['earlies_credit_year'].replace(2022,2000)
#数据转换
# data1['total_loan']=data1['total_loan'].apply(np.log)
# test1['total_loan']=test1['total_loan'].apply(np.log)
# data1['monthly_payment']=data1['monthly_payment'].apply(np.log)
# data1['early_return_amount']=(data1['early_return_amount']+2).apply(np.log)
# plt.figure(figsize=(20,20))
# data1['monthly_payment'].plot(kind = 'kde')
# plt.show()
#相关性查看
# plt.figure(figsize=(20,20))
# plt.scatter(data1['total_loan'],data1['monthly_payment'])
# plt.show()
# plt.figure(figsize=(20,20))
# plt.scatter(data1['class'],data1['interest'])
# plt.show()
#孤立森林test
# iforest=es.IsolationForest(n_estimators=100, max_samples='auto',
#                           contamination=0.0005,
#                           bootstrap=False,)
# index=set('')
# for i in data1.columns.values.tolist():
#     if (i != 'loan_id') and (i != 'user_id'):
#         tmp=pd.Series(iforest.fit_predict(data1[[i]]))
#         index.update(tmp[tmp==-1].index.values)
# print(len(index))
# data1=data1.drop(list(index))
#删除total_loan测试
data1=data1.sort_values(by=['isDefault'], ascending=True)
# data1=data1.drop('monthly_payment',axis=1)
# test1=test1.drop('monthly_payment',axis=1)
# data1=data1.drop('interest',axis=1)
# test1=test1.drop('interest',axis=1)
# data1=data1.drop('class',axis=1)
# data1=data1.drop('sub_class',axis=1)
# data1=data1.drop('loanTerm',axis=1)
# data1=data1.drop('scoring_mean',axis=1)
# data1=data1.drop('scoring_high',axis=1)
# data1=data1.drop('scoring_low',axis=1)
# data1=data1.drop('zero',axis=1)
#输出处理结果一
#data1.to_csv('before_normalization.csv',index=False)

#数字归一化，用Z-score进行归一化
test1.to_csv('after_preprocessing_test.csv')
data2:pd.DataFrame=(data1-data1.mean())/data1.std()
test2=(test1-test1.mean())/test1.std()

#特征选取
# rfc=es.RandomForestClassifier(n_estimators=60)
#
# feature_list=[]
# for i in data1.columns.values:
#     if(i!='isDefault'):
#         score = sklearn.model_selection.cross_val_score(rfc,data2[i].to_numpy().reshape(-1,1),data1['isDefault'] , scoring='roc_auc', cv=5)
#         feature_list.append(score.mean())
# tmp_columns:list=data1.columns.values.tolist()
# tmp_columns.remove('isDefault')
# feature:dict=dict(zip(tmp_columns,feature_list))
# feature=dict(sorted(feature.items(),key=lambda x:x[1],reverse=True))
# for i in feature:
#     print(i,"",feature[i])
data2=data2.drop('isDefault', axis=1)
# data2=pd.DataFrame(data2,columns=list(feature.keys()))
# test2=pd.DataFrame(test2,columns=list(feature.keys()))
# feature={k:v for k, v in feature.items() if v<0.5}
# data2=data2.drop(list(feature.keys()),axis=1)
# test2=test2.drop(list(feature.keys()),axis=1)
x = data2
y = data1['isDefault']
#使用EasyEnsemble做一次
#试一下随机森林
rfc=es.RandomForestClassifier(n_estimators=168,max_depth=75,random_state=1024)#class_weight={0:6,1:7}最优
eec=EasyEnsembleClassifier(base_estimator=rfc,n_estimators=20,random_state=1024)
brfc=BalancedRandomForestClassifier(n_estimators=168,random_state=1024)
rusb=RUSBoostClassifier(base_estimator=rfc,random_state=1024)
bbc=BalancedBaggingClassifier(base_estimator=rfc,random_state=1024)
# param = {"n_estimators": range(15,25)}
# gscv=GridSearchCV(estimator=eec,param_grid=param,scoring='roc_auc')
# gscv.fit(x,y)
# print(gscv.best_params_,gscv.best_score_)
# score=sklearn.model_selection.cross_val_score(rfc,x,y,scoring='roc_auc',cv=5)
# print('rfc',score,' ',score.mean())
# iforest=es.IsolationForest(n_estimators=100, max_samples='auto',
#                           contamination=0.1683,
#                           bootstrap=False,random_state=1024)
#print(sklearn.metrics.accuracy_score(y,x))
#试一下lgb
# model = lgb.LGBMClassifier(objective='binary',
#                            boosting_type='gbdt',
#                            tree_learner='serial',
#                            num_leaves=16,
#                            max_depth=4,
#                            learning_rate=0.01,
#                            n_estimators=10000,
#                            subsample=0.45,
#                            feature_fraction=0.5,
#                            reg_alpha=0.1,
#                            reg_lambda=0.5,
#                            random_state=2021,
#                            is_unbalance=True,
#                            num_thread=40,
#                            metric='auc')
# score=sklearn.model_selection.cross_val_score(model,x,y,scoring='roc_auc',cv=5)
# print('lgb',score,' ',score.mean())
def draw_auc(Y_val,y_val_predict_proba,model_name):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_val, y_val_predict_proba[:, 1])
    roc_auc_plt = sklearn.metrics.auc(fpr, tpr)
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    plt.plot(fpr, tpr, color='black', lw=1)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.text(0.5, 0.3, 'ROC curve (area = %0.3f)' % roc_auc_plt)
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig('../boxplot/{}_roc_eec.png'.format(model_name))
    plt.show()
#提交测试
roc_auc = []
accuracy=[]
roc_auc_brfc=[]
roc_auc_rusb=[]
roc_auc_bbc=[]
roc_auc_rfc=[]
ks=[]
kfold = StratifiedShuffleSplit(n_splits=5,random_state=1901)
for flod_id,(trn_idx, val_idx) in enumerate(kfold.split(x, y)):
    X_train = x.iloc[trn_idx]
    Y_train = y.iloc[trn_idx]

    X_val = x.iloc[val_idx]
    Y_val = y.iloc[val_idx]

    eec_model=eec.fit(X_train,Y_train)
    brfc_model=brfc.fit(X_train,Y_train)
    rusb_model=rusb.fit(X_train,Y_train)
    bbc_model=bbc.fit(X_train,Y_train)
    rfc_model = rfc.fit(X_train, Y_train)

    y_val_predict_proba = rfc_model.predict_proba(X_val)
    roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
    roc_auc_rfc.append(roc_score.mean())

    y_val_predict = rfc_model.predict(X_val)
    print('rfc/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
    print(roc_score)

    y_val_predict_proba=eec_model.predict_proba(X_val)
    roc_score=sklearn.metrics.roc_auc_score(Y_val,y_val_predict_proba[:,1])
    roc_auc.append(roc_score.mean())

    y_val_predict=eec_model.predict(X_val)
    print('eec/n',sklearn.metrics.classification_report(y_true=Y_val,y_pred=y_val_predict))
    print(roc_score)

    y_val_predict_proba = brfc_model.predict_proba(X_val)
    roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
    roc_auc_brfc.append(roc_score.mean())

    y_val_predict = brfc_model.predict(X_val)
    print('brfc/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
    print(roc_score)

    y_val_predict_proba = rusb_model.predict_proba(X_val)
    roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
    roc_auc_rusb.append(roc_score.mean())

    y_val_predict = rusb_model.predict(X_val)
    print('rusb/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
    print(roc_score)

    # test_y=eec_model.predict_proba(test2)
    # prediction['isDefault'] += test_y[:, 1] / (kfold.n_splits)

    del eec_model,X_train, Y_train, X_val, Y_val,brfc_model#,test_y
    gc.collect()
print('rfc',roc_auc_rfc,np.mean(roc_auc_rfc))
print('eec',roc_auc,np.mean(roc_auc))
print('brfc',roc_auc_brfc,np.mean(roc_auc_brfc))
print('rusb',roc_auc_rusb,np.mean(roc_auc_rusb))
# iforest_predict=iforest.fit_predict(x)
# iforest_predict=np.where(iforest_predict==-1,iforest_predict,0)
# iforest_predict=np.where(iforest_predict==0,iforest_predict,1)
# print('iforest/n',sklearn.metrics.classification_report(y_true=y,y_pred=iforest_predict))
# print('brfc',roc_auc_brfc,np.mean(roc_auc_brfc))
# print('rusb',roc_auc_rusb,np.mean(roc_auc_rusb))
# print('bbc',roc_auc_bbc,np.mean(roc_auc_bbc))
# print('rfc',roc_auc_rfc,np.mean(roc_auc_rfc))
prediction.columns = ['id', 'isDefault']
prediction.to_csv('submission.csv', index=False)#submission isdefault排序，data1特征选择，test1特征选择；submission1 isdefault排序，无特征选择；
# y_test.to_csv('submission.csv',index=0)#submission2 isdefault不排序，保留特征选择
#处理结果输出
# data2['isDefault']=data1['isDefault']
# data2.to_csv('after_preprocessing.csv')

