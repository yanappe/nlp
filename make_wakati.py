#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import pandas as pd
import MeCab

def processing(dirs):
    
    #記事のラベル
    article_label = []
    
    #記事
    article = []
    
    mecab = MeCab.Tagger()
    
    for i,dir_name in enumerate(list(dirs.values())):
        
        #ディクショナリ内のファイルをリスト化
        files = os.listdir('text/'+dir_name)
        
        for file in files:
            
            #記事のみ抽出
            if dir_name in file:
            
                with open('text/'+dir_name+'/'+file,'r') as f:
                    original = f.read()
                    
                
                # 半角数字0～9、小文字大文字a～ZなどURLに含まれる文字列を除去
                reg_text = re.sub(r'[0-9a-zA-Z]+', '', original)
                reg_text = re.sub(r'[:;/+\.-]', '', reg_text)
                
                #空欄と改行をなくす
                reg_text = re.sub(r'[\s\n]','',reg_text)
                
                #mecabで文章を分解
                lines = mecab.parse(reg_text)
                
                #分解された文章を行ごとにリスト化
                lines = lines.split('\n')
                
                #名詞を格納するリスト
                l = []
                
                for line in lines:
                    
                    #言葉とその情報を切り離す
                    line = line.split('\t') #タブ
                    
                    #EOS以外を判別
                    if len(line)==2 :
                        
                        l.append(line[0])  #名詞のみ抽出
                        
                #リスト中身を空白で連結
                l = ' '.join(l)
                
                #単語
                article.append(l)
                
                #記事の元
                article_label.append(i)
                
    df = pd.DataFrame({'ラベル':article_label,'文書':article,})
    
    df.to_pickle('data.pkl')


if __name__=='__main__':
    
    
    '''
    使用したデータ
    ライブドアコーパスデータ:https://www.rondhuit.com/download.html#ldcc
    '''
    
    dirs = {0:'dokujo-tsushin',1:'it-life-hack', 2:'kaden-channel', 3:'livedoor-homme',\
            4:'movie-enter', 5:'peachy', 6:'smax', 7:'sports-watch', 8:'topic-news'}
    
    processing(dirs)