#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold  #層化抽出法を用いたK-分割交差検証

def cv_method(model ,x_train, x_test, y_train, y_test, model_name ='not nn'):
    predict_val = []  #検証結果
    index_val = []  #検証結果のindex
    
    #foldの分け方を一定にする
    kf = StratifiedKFold(n_splits=4,random_state=0)
    
    #test_indexが交差検証における検証部分のデータ
    for train_index, val_index in kf.split(x_train,y_train):
        
        if model_name == 'nn':
            
            #ワンホットに変える
            y_train_onehot = np.array(pd.get_dummies(y_train[train_index]))
            
            y_val_onehot = np.array(pd.get_dummies(y_train[val_index]))
            
            batch_size = 128
            epochs =10
            
            #学習
            model.fit(x_train[train_index], y_train_onehot,\
                      batch_size = batch_size, epochs = epochs,\
                      verbose = 1, validation_data = (x_train[val_index],y_val_onehot))
            
            predicted_val = model.predict(x_train[val_index])
            
            #argmaxで処理
            predicted_val = np.argmax(predicted_val,axis=1)
            predict_val.append(predicted_val)
            
        
        else:
            
            #学習
            model.fit(x_train[train_index],y_train[train_index])
        
            #検証データで予測値を出力
            predicted_val = model.predict(x_train[val_index])
            predict_val.append(predicted_val)
        
        #検証データのindexを格納
        index_val.append(val_index)
    
    
    index_val = np.concatenate(index_val)  #検証データのindexを結合
    predict_val = np.concatenate(predict_val, axis = 0) #検証結果の配列を結合
    order = np.argsort(index_val)  #indexの並び替え
    
    #index順に並び替えられた検証結果の配列
    predict_train = predict_val[order]
    
    
    #テストデータの予測
    if model_name == 'nn':
        
        #ワンホットに変える
        y_train_onehot = np.array(pd.get_dummies(y_train))
        
        batch_size = 128
        epochs =10
        
        #学習
        model.fit(x_train, y_train_onehot,\
                  batch_size = batch_size, epochs = epochs,\
                  verbose = 1)
    
        #テスト結果
        predict_test = model.predict(x_test)
        
        #argmax処理
        predict_test = np.argmax(predict_test,axis=1)
        
        #正解率
        print(accuracy_score(y_test,predict_test))
    
        
    else:
        
        #テスト結果
        predict_test = model.fit(x_train,y_train).predict(x_test)
        
        #正解率
        print(accuracy_score(y_test,predict_test))
    
    #検証結果の配列とテスト結果の配列
    return predict_train, predict_test



if __name__ == '__main__':
    
    from models import model_rf
    from models import model_lgb
    from models import model_nn
    from models import model_svm
    
    from sklearn.metrics import accuracy_score   #正答率
    
    data_tfidf = pd.read_pickle('data_tfidf.pkl')
    
    X=np.array(data_tfidf.drop('ラベル', axis=1))
    y=np.array(data_tfidf['ラベル'])
    
    #stratifyでラベルの割合を調整
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state = 0)
    
    #ランダムフォレスト
    model1 = model_rf()
    predict_train_rf, predict_test_rf = cv_method(model1, x_train, x_test, y_train, y_test)
    
    #lightbm
    model2 = model_lgb()
    predict_train_lgb, predict_test_lgb = cv_method(model2, x_train, x_test, y_train, y_test)
    
    #ニューラルネット
    model3 = model_nn(X,y) #入力層と出力層を定める
    predict_train_nn, predict_test_nn = cv_method(model3,x_train, x_test, y_train, y_test, model_name ='nn')
    
    
    #アンサンブル学習
    train_ensemble_importance = pd.DataFrame({'rf':predict_train_rf,'lgb':predict_train_lgb,\
                                        'nn':predict_train_nn})
    
    train_ensemble_importance = np.array(train_ensemble_importance)
    
    test_ensemble_importance = pd.DataFrame({'rf':predict_test_rf,'lgb':predict_test_lgb,\
                                        'nn':predict_test_nn})
    
    test_ensemble_importance = np.array(test_ensemble_importance)
    
    
    model4 = model_svm()
    predict_train_svm, predict_test_svm = cv_method(model4, train_ensemble_importance,\
                                                    test_ensemble_importance,\
                                                    y_train, y_test)
    
    