#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  #単語に重み付け

def tfidf(data):
    df = pd.read_pickle(data)
    
    tv = TfidfVectorizer(min_df=0.01, max_df=0.5)
    data_tv = tv.fit_transform(df['文書'])
    data = data_tv.toarray()
    
    df_tv = pd.DataFrame(data)
    
    data = pd.concat([df_tv, df['ラベル']],axis=1)
    
    data.to_pickle('data_tfidf.pkl')

if __name__=='__main__':
    
    tfidf('data.pkl')
    