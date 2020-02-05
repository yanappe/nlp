#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from keras import models
from keras import layers
from sklearn import svm

#ランダムフォレスト
def model_rf():
    
    model = RandomForestClassifier(n_estimators=100,random_state = 0)
    
    return model
    
#lightgbm
def model_lgb():
    
    # 上記のパラメータでモデルを学習する
    model = lgb.LGBMClassifier(random_state = 0)
    
    return model
    
#ニューラルネット
def model_nn(X,y):
    
    input_number = X.shape[1]
    
    output_number = len(pd.Series(y).value_counts())  #クラス数
    
    model=models.Sequential()
    model.add(layers.Dense(412,activation='relu',input_shape=(input_number,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(412,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_number,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

#SVM
def model_svm():
    
    # 学習させる
    model = svm.SVC(gamma="scale")
    
    
    return model
