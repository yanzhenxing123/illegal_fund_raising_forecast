#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


#导入数据集
base_info = pd.read_csv('./train/train/base_info.csv')         # 企业基本信息
annual_report_info = pd.read_csv('./train/train/annual_report_info.csv')         # 年报基本信息
tax_info = pd.read_csv('./train/train/tax_info.csv')         #企业的纳税信息
change_info = pd.read_csv('./train/train/change_info.csv')         # 企业变更信息
news_info = pd.read_csv('./train/train/news_info.csv')         # 舆论新闻
other_info = pd.read_csv('./train/train/other_info.csv')         # 企业其他信息
entprise_info = pd.read_csv('./train/train/entprise_info.csv')         # 带标注的企业数据
entprise_evaluate = pd.read_csv('entprise_evaluate.csv')         # 测试


# In[3]:


print('base_info')
print(base_info.columns)
print('annual_report_info')
print(annual_report_info.columns)

print('entprise_info')
print(entprise_info.columns)

pd.to_datetime(tax_info['START_DATE'],format="%Y-%m-%d")

df_x = pd.DataFrame(entprise_info['id'])
df_y = pd.DataFrame(entprise_info['label'])
x_train, x_test,  y_train, y_test = train_test_split(df_x, df_y, test_size = 0.3, random_state = 2021)

data = pd.concat([x_train, x_test]).reset_index(drop=True)



def get_base_info_feature(df, base_info):

    off_data = base_info.copy()
    off_data_isnull_rate=off_data.isnull().sum()/len(off_data)
    big_null_name=off_data_isnull_rate[off_data_isnull_rate.values>=0.95].index
    base_info.drop(big_null_name,axis=1,inplace=True)

    base_info.fillna(-1, downcast = 'infer', inplace = True)
    #对时间的处理
    base_info['opfrom']=pd.to_datetime(base_info['opfrom'],format="%Y-%m-%d")  #把数据转换为时间类型
    base_info['pre_opfrom']=base_info['opfrom'].map(lambda x:x.timestamp() if x!=-1 else 0)   #将时间类型转换为时间戳
    base_info['opto']=pd.to_datetime(base_info['opto'],format='%Y-%m-%d')
    base_info['pre_opto']=base_info['opto'].map(lambda x:x.timestamp() if x!=-1 else 0)

    le=LabelEncoder()
    base_info['industryphy']=le.fit_transform(base_info['industryphy'].map(str))
    base_info['opscope']=le.fit_transform(base_info['opscope'].map(str))
    base_info['opform']=le.fit_transform(base_info['opform'].map(str))

    data = df.copy()
    data=pd.merge(data, base_info, on='id', how='left')
   # 行业类别基本特征
    key=['industryphy']
    prefixs = ''.join(key) + '_'
    #该行业有多少企业经营
    pivot=pd.pivot_table(data,index=key,values='id',aggfunc=lambda x:len(set(x)))
    pivot=pd.DataFrame(pivot).rename(columns={'id': prefixs+'different_id'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #行业广告经营特征
    key=['industryco','adbusign']
    #该行业有多少广告和不广告平均注册金
    pivot=pd.pivot_table(data,index=key,values='regcap',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'regcap': prefixs+'mean_regcap'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #细类行业特征
    key=['industryco']
    prefixs = ''.join(key) + '_'
    #该行业有多少企业经营
    pivot=pd.pivot_table(data,index=key,values='id',aggfunc=lambda x:len(set(x)))
    pivot=pd.DataFrame(pivot).rename(columns={'id': prefixs+'different_id'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #行业从业平均人数
    pivot=pd.pivot_table(data,index=key,values='empnum',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'empnum': prefixs+'mean_empnum'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)
    #行业从业人数最大
    pivot=pd.pivot_table(data,index=key,values='empnum',aggfunc=np.max)
    pivot=pd.DataFrame(pivot).rename(columns={'empnum': prefixs+'max_empnum'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #企业所有人数
    data['all_people']=list(map(lambda x,y,z : x+y+z ,data['exenum'],data['empnum'],data['parnum']))

    #企业实缴金额占注册多少
    data['rec/reg']=list(map(lambda x,y : x/y if y!=0 else 0,data['reccap'],data['regcap']))
    data.fillna(-1, downcast = 'infer', inplace = True)
    #企业没人共交多少
    data['mean_hand']=list(map(lambda x,y : x/y if y!=0 else 0,data['regcap'],data['all_people']))
    data.fillna(-1, downcast = 'infer', inplace = True)

    #经营范围(运动，材料)
    key=['opscope']
    prefixs = ''.join(key) + '_'
    #同样经营范围有那些企业
    pivot=pd.pivot_table(data,index=key,values='id',aggfunc=lambda x: len(set(x)))
    pivot=pd.DataFrame(pivot).rename(columns={'id': prefixs+'many_id'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #这种类型一个企业有多少从业人数
    pivot=pd.pivot_table(data,index=key,values='empnum',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'empnum': prefixs+'mean_empnum'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)
    # 这种类型共企业有多少合伙人
    pivot=pd.pivot_table(data,index=key,values='parnum',aggfunc=np.sum)
    pivot=pd.DataFrame(pivot).rename(columns={'parnum': prefixs+'sum_parnum'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)
    #这种类型一个企业有多少合伙人
    pivot=pd.pivot_table(data,index=key,values='parnum',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'parnum': prefixs+'mean_parnum'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)
    #这种范围平均注册金
    pivot=pd.pivot_table(data[data['regcap'].map(lambda x : x!=-1)],index=key,values='regcap',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'regcap': prefixs+'mean_ragcap'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)
    #这种范围最大和最小注册金
    pivot=pd.pivot_table(data[data['regcap'].map(lambda x : x!=-1)],index=key,values='regcap',aggfunc=np.max)
    pivot=pd.DataFrame(pivot).rename(columns={'regcap': prefixs+'max_ragcap'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #这种范围平均实缴金
    pivot=pd.pivot_table(data[data['reccap'].map(lambda x : x!=-1)],index=key,values='reccap',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'reccap': prefixs+'mean_raccap'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)
    #这种范围最大和最小实缴金
    pivot=pd.pivot_table(data[data['reccap'].map(lambda x : x!=-1)],index=key,values='reccap',aggfunc=np.max)
    pivot=pd.DataFrame(pivot).rename(columns={'reccap':prefixs+'max_raccap'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #企业类型
    key=['enttype']
    prefixs = ''.join(key) + '_'
    #企业类型有几个小类
    pivot=pd.pivot_table(data,index=key,values='enttypeitem',aggfunc=lambda x:len(set(x)))
    pivot=pd.DataFrame(pivot).rename(columns={'enttypeitem':
                                              prefixs+'different_item'}).reset_index()
    data = pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1, downcast = 'infer', inplace = True)

    #排序特征
    key=['sort']
    prefixs = ''.join(key) + '_'
    #行业类别注册金正反序
    data[prefixs+'industryphy_regcap_postive']=data.groupby('industryphy')['regcap'].rank(ascending=True)
    data[prefixs+'industryphy_regcap_nagative']=data.groupby('industryphy')['regcap'].rank(ascending=False)
    #行业类别投资金金正反序
    data[prefixs+'industryphy_reccap_postive']=data.groupby('industryphy')['reccap'].rank(ascending=True)
    data[prefixs+'industryphy_reccap_nagative']=data.groupby('industryphy')['reccap'].rank(ascending=False)

     #企业类型业注册金正反序
    data[prefixs+'enttype_regcap_postive']=data.groupby('enttype')['regcap'].rank(ascending=True)
    data[prefixs+'enttype_regcap_nagative']=data.groupby('enttype')['regcap'].rank(ascending=False)
    #企业类型投资金金正反序
    data[prefixs+'enttype_reccap_postive']=data.groupby('enttype')['reccap'].rank(ascending=True)
    data[prefixs+'enttype_reccap_nagative']=data.groupby('enttype')['reccap'].rank(ascending=False)

    #经营限期注册金正反序
    data[prefixs+'opfrom_regcap_postive']=data.groupby('pre_opfrom')['regcap'].rank(ascending=True)
    data[prefixs+'opfrom_regcap_negative']=data.groupby('pre_opfrom')['regcap'].rank(ascending=False)
    #经营限起投资金金正反序
    data[prefixs+'opfrom_recap_postive']=data.groupby('pre_opfrom')['reccap'].rank(ascending=True)
    data[prefixs+'opfrom_reccap_negative']=data.groupby('pre_opfrom')['reccap'].rank(ascending=False)

    #经营限期☞注册金正反序
    data[prefixs+'opto_regcap_postive']=data.groupby('pre_opto')['regcap'].rank(ascending=True)
    data[prefixs+'opto_regcap_negative']=data.groupby('pre_opto')['regcap'].rank(ascending=False)
    # #经营限止投资金金正反序
    # data[prefixs+'opto_recap_postive']=data.groupby('pre_opto')['reccap'].rank(ascending=True)
    data[prefixs+'opto_reccap_negative']=data.groupby('pre_opto')['reccap'].rank(ascending=False)

    #enttypegb注册金正反序
    data[prefixs+'enttypegb_regcap_postive']=data.groupby('enttypegb')['regcap'].rank(ascending=True)
    data[prefixs+'enttypegb_regcap_negative']=data.groupby('enttypegb')['regcap'].rank(ascending=False)
    #enttypegb投资金金正反序
    data[prefixs+'enttypegb_recap_postive']=data.groupby('enttypegb')['reccap'].rank(ascending=True)
    data[prefixs+'enttypegb_reccap_negative']=data.groupby('enttypegb')['reccap'].rank(ascending=False)

    # #sdbusign注册金正反序
    # data[prefixs+'adbusign_regcap_postive']=data.groupby('adbusign')['regcap'].rank(ascending=True)
    # data[prefixs+'adbusign_regcap_negative']=data.groupby('adbusign')['regcap'].rank(ascending=False)
    # #enttypegb投资金金正反序
    data[prefixs+'adbusign_recap_postive']=data.groupby('adbusign')['reccap'].rank(ascending=True)
    # data[prefixs+'adbusign_reccap_negative']=data.groupby('adbusign')['reccap'].rank(ascending=False)

    return data

data = get_base_info_feature(data, base_info)
# x_train = get_base_info_feature(x_train, base_info)
# x_test = get_base_info_feature(x_test, base_info)


# In[13]:


def get_annual_report_info_feature(df, feat):
    off_data=feat.copy()
    off_data_isnull_rate=off_data.isnull().sum()/len(off_data)
    big_null_name=off_data_isnull_rate[off_data_isnull_rate.values>=0.9].index
    feat.drop(big_null_name,axis=1,inplace=True)
    feat.fillna(-1,downcast = 'infer', inplace = True)
    #企业年报特征
    #企业

    data = df.copy()
    key=['id']
    prefixs = ''.join(key) + '_'
    #企业在几年内是否变更状态
    pivot=pd.pivot_table(feat,index=key,values='STATE',aggfunc=lambda x:len(set(x)))
    pivot=pd.DataFrame(pivot).rename(columns={'STATE':prefixs+'many_STATE'}).reset_index()
    data=pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1,downcast = 'infer', inplace = True)
    #企业资金总额
    pivot=pd.pivot_table(feat,index=key,values='FUNDAM',aggfunc=np.sum)
    pivot=pd.DataFrame(pivot).rename(columns={'FUNDAM':prefixs+'sum_FUNDAM'}).reset_index()
    data=pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1,downcast = 'infer', inplace = True)
    #企业从业人数
    pivot=pd.pivot_table(feat,index=key,values='EMPNUM',aggfunc=np.sum)
    pivot=pd.DataFrame(pivot).rename(columns={'EMPNUM':prefixs+'sum_EMPNUM'}).reset_index()
    data=pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1,downcast = 'infer', inplace = True)
    #企业有几年公布了从业人数
    pivot=pd.pivot_table(feat[feat['EMPNUMSIGN'].map(lambda x: x==1)],index=key,values='EMPNUM',aggfunc=len)
    pivot=pd.DataFrame(pivot).rename(columns={'EMPNUM':prefixs+'gongshi_many_EMPNUM '}).reset_index()
    data=pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1,downcast = 'infer', inplace = True)
    #企业有几年是开业
    pivot=pd.pivot_table(feat[feat['BUSSTNAME'].map(lambda x: x=='开业')],index=key,values='BUSSTNAME',aggfunc=len)
    pivot=pd.DataFrame(pivot).rename(columns={'BUSSTNAME':prefixs+'开业_many_year '}).reset_index()
    data=pd.merge(data, pivot, on=key, how='left')
    data.fillna(-1,downcast = 'infer', inplace = True)

    return data

data = get_annual_report_info_feature(data, annual_report_info)
train = data[:x_train.shape[0]].reset_index(drop=True)
test = data[x_train.shape[0]:].reset_index(drop=True)


def get_model(train_x,train_y,valid_x,valid_y, my_type='lgb'):
    if my_type == 'lgb':
        params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_error',
                'num_leaves': 64,
                'max_depth':7,
                'learning_rate': 0.02,
                'feature_fraction': 0.85,
                'feature_fraction_seed':2021,
                'bagging_fraction': 0.85,
                'bagging_freq': 5,
                'bagging_seed':2021,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.5,
                'lambda_l2': 1.2,
                'verbose': -1
            }
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)
        model = lgb.train(
            params,
            train_set = dtrain,
            num_boost_round=10000,
            valid_sets = [dtrain, dvalid],
            verbose_eval=100,
            early_stopping_rounds=600,
    #         categorical_feature=cat_cols,
        )
    elif my_type == 'xgb':
        params = {'booster':'gbtree', #线性模型效果不如树模型
              'objective':'binary:logistic',
              'eval_metric':'auc',
              'silent':1, #取0时会输出一大堆信息
              'eta':0.01, #学习率典型值为0.01-0.2
              'max_depth':7, #树最大深度，典型值为3-10，用来避免过拟合
              'min_child_weight':5, #默认取1，用于避免过拟合，参数过大会导致欠拟合
              'gamma':0.2, #默认取0，该参数指定了节点分裂所需的小损失函数下降值
              'lambda':1, #默认取1.权重的L2正则化项
              'colsample_bylevel':0.7,
              'colsample_bytree':0.8, #默认取1，典型值0.5-1，用来控制每棵树随机采样的比例
              'subsample':0.8, #默认取1，典型值0.5-1，用来控制对于每棵树，随机采样的比例
              'scale_pos_weight':1 #在各类样本十分不平衡时，设定该参数为一个正值，可使算法更快收敛
              }
        dtrain = xgb.DMatrix(train_x, label = train_y)
#         watchlist = [(dtrain,'train')] #列出每次迭代的结果
        model = xgb.train(params,dtrain,num_boost_round = 1200)

    elif my_type == 'cat':

        model = CatBoostClassifier(
                 iterations=5000,
                 max_depth=10,
                 learning_rate=0.07,
                 l2_leaf_reg=9,
                 random_seed=2018,
                 fold_len_multiplier=1.1,
                 early_stopping_rounds=100,
                 use_best_model=True,
                 loss_function='Logloss',
                 eval_metric='AUC',
                 verbose=100)

        model.fit(train_x,train_y,eval_set=[(train_x, train_y),(valid_x, valid_y)], plot=True)

    return model




from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score

result = pd.DataFrame()
model_type = ['cat', 'lgb', 'xgb']
for my_type in model_type:
    KF = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    features = [i for i in train.columns if i not in ['id','dom', 'opfrom', 'opto', 'oploc']]
    oof = np.zeros(len(train))
    predictions = np.zeros((len(test)))
    # 特征重要性
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})

#     my_type='cat'
    # 五折交叉验证
    for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y_train.values)):
        print("===================================fold_{}===================================".format(fold_))
        print('trn_idx:',trn_idx)
        print('val_idx:',val_idx)
        train_x, train_y = train.iloc[trn_idx][features], y_train.iloc[trn_idx]
        valid_x, valid_y = train.iloc[val_idx][features], y_train.iloc[val_idx]

        model = get_model(train_x, train_y, valid_x, valid_y, my_type)
        if my_type == 'lgb':
            feat_imp_df['imp'] += model.feature_importance() / 5
            oof[val_idx] = model.predict(valid_x, num_iteration=model.best_iteration)
            predictions[:] += model.predict(test[features], num_iteration=model.best_iteration)
        elif my_type == 'xgb':
            oof[val_idx] = model.predict(xgb.DMatrix(valid_x))
            predictions[:] += model.predict(xgb.DMatrix(test[features]))
        elif my_type == 'cat':
            oof[val_idx] = model.predict_proba(valid_x)[:,1]
            predictions[:] += model.predict_proba(test[features])[:,1]

    print("AUC score: {}".format(roc_auc_score(y_train, oof)))
    print("F1 score: {}".format(f1_score(y_train, [1 if i >= 0.5 else 0 for i in oof])))
    print("Precision score: {}".format(precision_score(y_train, [1 if i >= 0.5 else 0 for i in oof])))
    print("Recall score: {}".format(recall_score(y_train, [1 if i >= 0.5 else 0 for i in oof])))

    result[my_type] = predictions / 5


result['label'] = result[['cat', 'lgb', 'xgb']].sum(axis=1) /3

result['label'].describe()



feat_imp_df.sort_values(by='imp', ascending=False)

print("AUC score: {}".format(roc_auc_score(y_test, result['label'])))
print("F1 score: {}".format(f1_score(y_test, [1 if i >= 0.55 else 0 for i in result['label']])))
print("Precision score: {}".format(precision_score(y_test, [1 if i >= 0.55 else 0 for i in result['label']])))
print("Recall score: {}".format(recall_score(y_test, [1 if i >= 0.55 else 0 for i in result['label']])))

x_test['label'] = predictions / 5

