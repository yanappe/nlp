# livedoorニュースコーパスの記事カテゴリ判別器

## 概要

livedoorコーパス
(https://www.rondhuit.com/download.html)
から、記事を取得し、記事カテゴリ判別をアンサンブル学習によって行いました。
初めに、MeCabで形態素解析をした後、tfidfの処理をしました。それらを、train_test_splitで記事カテゴリの割合が均等になるよう分けた後、訓練データで交差検証を行い、テストデータで予測精度を算出しました。用いる機械学習モデルは、ランダムフォレスト、lightgbm、ニューラルネットワークそれぞれで、判別しました。
最後に3つの出力値を特徴量として使い、SVMで予測精度を算出しました。

## 予測精度

RandomForest : 90.7%

lightgbm : 93.8%

NN : 91.4%

SVMを使ったアンサンブル結果 : 91.3%

まだ改善の余地あり。。。。

## 参考文献

1.機械学習のための「前処理」入門

2.kaggleで勝つデータ分析の技術

## 使用した主なパッケージ

python 3.7.3

scikit-learn 0.21.3

keras 2.2.4

lightgbm 2.3.0

numpy 1.16.4

pandas 0.24.2