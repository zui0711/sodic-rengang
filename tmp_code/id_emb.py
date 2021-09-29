import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
df_person = pd.read_csv('data/trainset/person.csv')
df_person_cv = pd.read_csv('data/trainset/person_cv.csv')
df_person_job_hist = pd.read_csv('data/trainset/person_job_hist.csv')
df_person_pro_cert = pd.read_csv('data/trainset/person_pro_cert.csv')
df_person_pro_cert['PRO_CERT_DSP'] = df_person_pro_cert['PRO_CERT_DSP'].map(lambda x: x.replace('CET-4', '').replace('CET-6', ''))

df_person_project = pd.read_csv('data/trainset/person_project.csv')
df_recruit = pd.read_csv('data/trainset/recruit.csv')

df_train = pd.read_csv('data/trainset/recruit_folder.csv')
df_test = pd.read_csv('data/testset/recruit_folder.csv')

df_person = pd.merge(df_person, df_person_cv, on='PERSON_ID', how='left')
df_person = pd.merge(df_person, df_person_job_hist, on='PERSON_ID', how='left')
df_person = pd.merge(df_person, df_person_pro_cert, on='PERSON_ID', how='left')
df_person = pd.merge(df_person, df_person_project, on='PERSON_ID', how='left')

df_train = pd.merge(df_train, df_recruit, on='RECRUIT_ID', how='left')
df_train = pd.merge(df_train, df_person, on='PERSON_ID', how='left').drop_duplicates(subset=['PERSON_ID', 'RECRUIT_ID']).reset_index(drop=True)

df_test = pd.merge(df_test, df_recruit, on='RECRUIT_ID', how='left')
df_test = pd.merge(df_test, df_person, on='PERSON_ID', how='left')#.drop_duplicates(subset=['PERSON_ID', 'RECRUIT_ID']).reset_index(drop=True)

df = pd.concat([df_train, df_test]).reset_index(drop=True)
sent = []
num = []
for rid in tqdm(df['RECRUIT_ID'].unique()):
    d = df.loc[df['RECRUIT_ID']==rid, ['RECRUIT_ID', 'PERSON_ID']].drop_duplicates(subset=['RECRUIT_ID', 'PERSON_ID']).reset_index(drop=True)
    if 316816964 in d:
        print('!!!', rid)
    _sent = list(map(str, d['PERSON_ID'].values))
    sent.append(_sent)
    num.append(len(_sent))
print(max(num))

model = Word2Vec(sentences=sent, vector_size=8, window=16, min_count=1, workers=1)
model.save("word2vec.model")
# model = Word2Vec.load('word2vec.model')
emb = []
pid = df[['PERSON_ID']].drop_duplicates().reset_index(drop=True)
for id in tqdm(pid['PERSON_ID']):
    # print(id)
    emb.append(model.wv[str(id)])
emb = pd.DataFrame(emb)
emb.columns = ['PERSON_ID_emb_%d'%i for i in range(8)]
emb['PERSON_ID'] = pid['PERSON_ID'].values
emb.to_csv('PERSON_ID_emb.csv', index=False)
