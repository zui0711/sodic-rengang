import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import lightgbm as lgb
from matplotlib.pyplot import plot, show
import time

df_person = pd.read_csv('data/trainset/person.csv')
df_person['HIGHEST_EDU'] = df_person['HIGHEST_EDU'].map(
    {"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
df_person.rename(columns={'MAJOR': 'PERSON_MAJOR'}, inplace=True)
df_person['PERSON_MAJOR'] = df_person['PERSON_MAJOR'].map(
    lambda x: x.replace('【', '').replace('】', '') if type(x) != float else x)

df_person_cv = pd.read_csv('data/trainset/person_cv.csv')
df_person_job_hist = pd.read_csv('data/trainset/person_job_hist.csv')
df_person_pro_cert = pd.read_csv('data/trainset/person_pro_cert.csv')
df_person_pro_cert['PRO_CERT_DSP'] = df_person_pro_cert['PRO_CERT_DSP'].map(lambda x: x.replace('CET-4', '').replace('CET-6', ''))

df_person_project = pd.read_csv('data/trainset/person_project.csv')
df_recruit = pd.read_csv('data/trainset/recruit.csv')
df_recruit['LOWER_EDU'] = df_recruit['LOWER_EDU'].map(
    {"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
df_recruit['WORK_YEARS_RANGE'] = df_recruit['WORK_YEARS_RANGE'].map(
    {"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5})
df_recruit.rename(columns={'MAJOR': 'RECRUIT_MAJOR'}, inplace=True)
df_recruit['RECRUIT_MAJOR'] = df_recruit['RECRUIT_MAJOR'].map(
    lambda x: x.replace('【', '').replace('】', '') if type(x) != float else x)

df_train = pd.read_csv('data/trainset/recruit_folder.csv')
df_test = pd.read_csv('data/testset/recruit_folder.csv')

df_person = pd.merge(df_person, df_person_cv, on='PERSON_ID', how='left')
df_person = pd.merge(df_person, df_person_job_hist, on='PERSON_ID', how='left')
df_person = pd.merge(df_person, df_person_pro_cert, on='PERSON_ID', how='left')
df_person = pd.merge(df_person, df_person_project, on='PERSON_ID', how='left')

df_train = pd.merge(df_train, df_recruit, on='RECRUIT_ID', how='left')
df_train = pd.merge(df_train, df_person, on='PERSON_ID', how='left').drop_duplicates(subset=['PERSON_ID', 'RECRUIT_ID']).reset_index(drop=True)

df_test = pd.merge(df_test, df_recruit, on='RECRUIT_ID', how='left')
df_test = pd.merge(df_test, df_person, on='PERSON_ID', how='left')

df_train['EDU'] = (df_train['HIGHEST_EDU'] - df_train['LOWER_EDU']).map(lambda x: 1 if x>=0 else 0)
df_test['EDU'] = (df_test['HIGHEST_EDU'] - df_test['LOWER_EDU']).map(lambda x: 1 if x>=0 else 0)

df_train['YEARS'] = (df_train['WORK_YEARS'] - df_train['WORK_YEARS_RANGE']).map(lambda x: 1 if x>=0 else 0)
df_test['YEARS'] = (df_test['WORK_YEARS'] - df_test['WORK_YEARS_RANGE']).map(lambda x: 1 if x>=0 else 0)

feats = ['PERSON_TYPE_CODE', 'AGE', 'AVAILABLE_IN_DAYS',
         'HIGHEST_EDU', 'LOWER_EDU', 'EDU',
         'WORK_YEARS', 'WORK_YEARS_RANGE', 'YEARS',
         'PERSON_ID', 'RECRUIT_ID',]

df_tmp = pd.concat([df_train, df_test])
for name in ['PERSON_TYPE', 'JOB_TITLE', 'PERSON_MAJOR', 'LOCATION',
             'GENDER', 'RECRUIT_MAJOR', 'CURR_LOC',
             'LOCATION_x', 'LOCATION_y',
             # 'INDUSTRY_x', 'INDUSTRY_y',
             # 'POSITION_x', 'POSITION_y',
             # 'PRO_CERT_DSP'
             ]:
    le = LabelEncoder()
    df_tmp[name] = le.fit_transform(df_tmp[name])
    df_train[name] = le.transform(df_train[name]).astype('float')
    df_test[name] = le.transform(df_test[name]).astype('float')
    feats.append(name)

df_train['LOC'] = (df_train['LOCATION'] - df_train['LOCATION_x']).map(lambda x: 1 if x==0 else 0)
df_test['LOC'] = (df_test['LOCATION'] - df_test['LOCATION_x']).map(lambda x: 1 if x==0 else 0)
feats.append('LOC')
#
df_train['MAJ'] = (df_train['PERSON_MAJOR'] - df_train['RECRUIT_MAJOR']).map(lambda x: 1 if x==0 else 0)
df_test['MAJ'] = (df_test['PERSON_MAJOR'] - df_test['RECRUIT_MAJOR']).map(lambda x: 1 if x==0 else 0)
feats.append('MAJ')

job_feat_names = ['LABEL']
job_feats = []
[job_feats.extend([x + '_job_mean']) for x in job_feat_names]

df = pd.concat([df_train, df_test.drop_duplicates(subset=['PERSON_ID', 'RECRUIT_ID']).reset_index(drop=True)])
tmp = df[['RECRUIT_ID', 'AGE', 'GENDER']].groupby(['RECRUIT_ID']).agg({'mean'}).reset_index()
tmp.columns = ["RECRUIT_ID", "job_AGE_mean", "job_GENDER_mean"]
df_train = pd.merge(df_train, tmp, on='RECRUIT_ID', how='left')
df_test = pd.merge(df_test, tmp, on='RECRUIT_ID', how='left')
feats = feats + ["job_AGE_mean", "job_GENDER_mean"]


def target_feats(train, test):
    tmp_person = train[['PERSON_ID', 'LABEL']].groupby('PERSON_ID').agg({"mean"}).reset_index()
    tmp_person.columns = ["PERSON_ID", "LABEL_person_mean"]
    tmp_job = train[['RECRUIT_ID'] + job_feat_names].groupby('RECRUIT_ID').agg({'mean'}).reset_index()
    tmp_job.columns = ["RECRUIT_ID"] + job_feats

    ret = pd.merge(test, tmp_person, on='PERSON_ID', how='left')
    ret = pd.merge(ret, tmp_job, on='RECRUIT_ID', how='left')

    for name in ['GENDER', 'AGE']:
        tmp = train[[name, 'LABEL']].groupby(name).agg({'mean', 'std'}).reset_index()
        tmp.columns = [name, 'LABEL_'+name+'_mean', 'LABEL_'+name+'_std']
        ret = pd.merge(ret, tmp, on=name, how='left')

    for name in ['GENDER', 'AGE']:
        tmp = train[[name, 'RECRUIT_ID', 'LABEL']].groupby(['LABEL', 'RECRUIT_ID']).agg({'mean', 'std'}).reset_index()
        tmp.columns = ['LABEL', 'RECRUIT_ID', 'LABEL_RECRUIT_'+name+'_mean', 'LABEL_RECRUIT_'+name+'_std']
        ret = pd.merge(ret, tmp.loc[tmp['LABEL']==1,
                                    ['RECRUIT_ID', 'LABEL_RECRUIT_'+name+'_mean', 'LABEL_RECRUIT_'+name+'_std']],
                       on=['RECRUIT_ID'], how='left')

    return ret


print(df_train['LABEL'].sum() / len(df_train))
print(df_test.shape)
df_test = target_feats(df_train, df_test)
print(df_test.shape)
params = {
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'verbose': -1,
    'is_unbalance': True,
    'seed': 2021,
    'n_jobs': -1,
}

fold_num = 5
skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=2021)
oof = np.zeros(len(df_train))
y = 0
for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train[feats], df_train['LABEL'])):
    tmp = df_train.loc[trn_idx].reset_index(drop=True)
    df_v = target_feats(tmp, df_train.loc[val_idx].reset_index(drop=True))
    df_t = None
    for i in range(4):
        df1 = tmp.loc[tmp.index % 4 != i]
        df2 = tmp.loc[tmp.index % 4 == i]
        df_t = pd.concat([df_t, target_feats(df1, df2)], ignore_index=True)

    n_feats = feats + ['LABEL_person_mean'] + job_feats + ['LABEL_GENDER_mean', 'LABEL_GENDER_std', 'LABEL_AGE_mean', 'LABEL_AGE_std',
                                                           'LABEL_RECRUIT_GENDER_mean', 'LABEL_RECRUIT_GENDER_std',
                                                           'LABEL_RECRUIT_AGE_mean', 'LABEL_RECRUIT_AGE_std',]

    train_x = df_t[n_feats]
    train_y = df_t['LABEL']
    val_x = df_v[n_feats]
    val_y = df_v['LABEL']
    print('-------------------------FOLD %d'%fold)
    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)

    train_dataset = lgb.Dataset(train_x, train_y)
    val_dataset = lgb.Dataset(val_x, val_y)

    model = lgb.train(params, train_dataset, valid_sets=val_dataset, num_boost_round=3000, early_stopping_rounds=50,
                      verbose_eval=1500)
    # lgb.plot_importance(model, importance_type='gain')
    # show()
    oof[val_idx] = model.predict(df_v[n_feats])
    y += model.predict(df_test[n_feats]) / fold_num

print(log_loss(df_train['LABEL'], oof),
      roc_auc_score(df_train['LABEL'], oof),
      f1_score(df_train['LABEL'], list(map(lambda x: 1 if x > 0.6 else 0, oof))))
df_test['LABEL'] = y
df_test = df_test[['RECRUIT_ID', 'PERSON_ID', 'LABEL']].groupby(['RECRUIT_ID', 'PERSON_ID']).agg('max').reset_index()

print(df_test.shape)
df_test['LABEL'] = df_test['LABEL'].map(lambda x: 1 if x > 0.6 else 0)
print(df_test.shape)
print(df_test['LABEL'].sum())
plot(df_test['LABEL'].values[:1000])
show()
df_test[['RECRUIT_ID', 'PERSON_ID', 'LABEL']].to_csv(time.strftime('ans/ans_pred.csv'), index=False)
