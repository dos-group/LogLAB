from tqdm import tqdm
import pandas as pd

def label_data(df, timerange):
    fault_times = set(df[df['Label']==1]['datetime'].unique())
    mask = [True for _ in range(len(df))]
    for t in tqdm(fault_times):
        m = ((df['datetime'] >= (t + pd.Timedelta(milliseconds=timerange))) | 
             (df['datetime'] <= (t - pd.Timedelta(milliseconds=timerange))))
        mask &= m
    df['Localize_Label'] = mask*1
    df['Localize_Label'].replace({0: 1, 1: 0}, inplace=True)
    return df


def get_vocab_size(tokenized_lists):
    vocabs = []
    for t_seq in tqdm(tokenized_lists):
        for t in t_seq:
            vocabs.append(t)
    vocabs = set(vocabs)
    vocabs = vocabs.difference(set({0,1,2,3,4}))
    return len(vocabs) + 5