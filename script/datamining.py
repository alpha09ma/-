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
#导入数据集，并删去异常值以及进行部分修正
data1=pd.read_csv('./feature.csv')
data1=data1.drop(data1[data1['debt_loan_ratio'].max()==data1['debt_loan_ratio']].index.values)
data1=data1.drop(data1[data1['debt_loan_ratio'].max()==data1['debt_loan_ratio']].index.values)
data1=data1.drop(data1[data1['recircle_b']>500000].index.values)
data1=data1.drop(data1[data1['house_exist'].max()==data1['house_exist']].index.values)
data1['earlies_credit_year']=data1['earlies_credit_year'].replace(2022,2000)
data1=data1.sort_values(by=['isDefault'], ascending=True)
#数字归一化，用Z-score进行归一化
data2:pd.DataFrame=(data1-data1.mean())/data1.std()

data2=data2.drop('isDefault', axis=1)
x = data2
y = data1['isDefault']
#导入各模型
rfc=es.RandomForestClassifier(n_estimators=168,max_depth=75,random_state=1024)#class_weight={0:6,1:7}最优
eec=EasyEnsembleClassifier(base_estimator=rfc,n_estimators=20,random_state=1024)
brfc=BalancedRandomForestClassifier(n_estimators=168,random_state=1024)
rusb=RUSBoostClassifier(base_estimator=rfc,random_state=1024)
bbc=BalancedBaggingClassifier(base_estimator=rfc,random_state=1024)
#选取参数，由于耗时过长正式预测时注释
# param = {"n_estimators": range(15,25)}
# gscv=GridSearchCV(estimator=eec,param_grid=param,scoring='roc_auc')
# gscv.fit(x,y)
# print(gscv.best_params_,gscv.best_score_)
def draw_auc(fpr,tpr,model_name):
    k=0
    model_name_list=['rfc','eec','brfc','rusb']
    color_list=['crimson','orange','gold','mediumseagreen',]
    for i,j in zip(fpr,tpr):
        roc_auc_plt = sklearn.metrics.auc(i, j)
        plt.plot(i, j, color=color_list[k], lw=1,label='{0} (AUC={1:.3f})'.format(model_name_list[k], roc_auc_plt))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.legend(loc='lower right', fontsize=10)
        k+=1
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig('../boxplot/{}_roc_auc.png'.format(model_name))
    plt.show()
#提交测试
roc_auc = []
roc_auc_brfc=[]
roc_auc_rusb=[]
roc_auc_rfc=[]
fpr_list={'rfc':0,'brfc':0,'eec':0,'rusb':0}
tpr_list={'rfc':0,'brfc':0,'eec':0,'rusb':0}
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
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_val, y_val_predict_proba[:, 1])
    fpr_list['rfc']=fpr
    tpr_list['rfc']=tpr

    y_val_predict_proba=eec_model.predict_proba(X_val)
    roc_score=sklearn.metrics.roc_auc_score(Y_val,y_val_predict_proba[:,1])
    roc_auc.append(roc_score.mean())
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_val, y_val_predict_proba[:, 1])

    fpr_list['eec'] = fpr
    tpr_list['eec'] = tpr

    y_val_predict=eec_model.predict(X_val)
    print('eec/n',sklearn.metrics.classification_report(y_true=Y_val,y_pred=y_val_predict))
    print(roc_score)

    y_val_predict_proba = brfc_model.predict_proba(X_val)
    roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
    roc_auc_brfc.append(roc_score.mean())

    y_val_predict = brfc_model.predict(X_val)
    print('brfc/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
    print(roc_score)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_val, y_val_predict_proba[:, 1])
    fpr_list['brfc'] = fpr
    tpr_list['brfc'] = tpr

    y_val_predict_proba = rusb_model.predict_proba(X_val)
    roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
    roc_auc_rusb.append(roc_score.mean())

    y_val_predict = rusb_model.predict(X_val)
    print('rusb/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
    print(roc_score)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_val, y_val_predict_proba[:, 1])
    fpr_list['rusb'] = fpr
    tpr_list['rusb'] = tpr

    fpr_list = list(fpr_list.values())
    tpr_list = list(tpr_list.values())
    draw_auc(fpr_list, tpr_list, 'mix{}'.format(flod_id))
    fpr_list = {'rfc': 0, 'brfc': 0, 'eec': 0, 'rusb': 0}
    tpr_list = {'rfc': 0, 'brfc': 0, 'eec': 0, 'rusb': 0}
    del eec_model,X_train, Y_train, X_val, Y_val#,brfc_model,rfc_model,rusb_model
    gc.collect()
print('rfc',roc_auc_rfc,np.mean(roc_auc_rfc))
print('eec',roc_auc,np.mean(roc_auc))
print('brfc',roc_auc_brfc,np.mean(roc_auc_brfc))
print('rusb',roc_auc_rusb,np.mean(roc_auc_rusb))
#特征选取，由于结果较差最后模型比较时注释掉
# rfc=es.RandomForestClassifier(n_estimators=60)
#
# feature_list=[]
# for i in data2.columns.values:
#     score = sklearn.model_selection.cross_val_score(rfc,data2[i].to_numpy().reshape(-1,1),data1['isDefault'] , scoring='roc_auc', cv=5)
#     feature_list.append(score.mean())
# tmp_columns:list=data2.columns.values.tolist()
# feature:dict=dict(zip(tmp_columns,feature_list))
# feature=dict(sorted(feature.items(),key=lambda x:x[1],reverse=True))
# for i in feature:
#     print(i,"",feature[i])
# data2=pd.DataFrame(data2,columns=list(feature.keys()))
# feature={k:v for k, v in feature.items() if v<0.5}
# data2=data2.drop(list(feature.keys()),axis=1)
# x=data2
# y=data1['isDefault']
# roc_auc = []
# roc_auc_rfc = []
# kfold = StratifiedShuffleSplit(n_splits=5, random_state=1901)
# for flod_id, (trn_idx, val_idx) in enumerate(kfold.split(x, y)):
#     X_train = x.iloc[trn_idx]
#     Y_train = y.iloc[trn_idx]
#
#     X_val = x.iloc[val_idx]
#     Y_val = y.iloc[val_idx]
#
#     eec_model = eec.fit(X_train, Y_train)
#     rfc_model = rfc.fit(X_train, Y_train)
#
#     y_val_predict_proba = rfc_model.predict_proba(X_val)
#     roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
#     roc_auc_rfc.append(roc_score.mean())
#
#     y_val_predict = rfc_model.predict(X_val)
#     print('rfc/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
#     print(roc_score)
#
#     y_val_predict_proba = eec_model.predict_proba(X_val)
#     roc_score = sklearn.metrics.roc_auc_score(Y_val, y_val_predict_proba[:, 1])
#     roc_auc.append(roc_score.mean())
#
#     y_val_predict = eec_model.predict(X_val)
#     print('eec/n', sklearn.metrics.classification_report(y_true=Y_val, y_pred=y_val_predict))
#     print(roc_score)
# print('rfc',roc_auc_rfc,np.mean(roc_auc_rfc))
# print('eec',roc_auc,np.mean(roc_auc))