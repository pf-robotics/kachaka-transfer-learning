### PCでの環境準備

* 適当な手元のPCで以下のコマンドを実行し、必要なpythonモジュールをインストールして下さい。

```
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### データセット準備

* [ノートブック](https://github.com/pf-robotics/kachaka-api/blob/master/python/demos/save_object_detection_features.ipynb)に従い、カチャカ体内で必要なデータを取得して下さい。

* 手元のPCで [アノテーションツール](https://github.com/jsbroks/coco-annotator.git)を立ち上げ、お好きなブラウザで http://localhost:5000/ でアクセスします(※ ポート被りがある場合はポート番号が変わります)。
```
git clone https://github.com/jsbroks/coco-annotator.git
cd coco-annotator && sudo docker-compose up
```

* ユーザー作成・ログイン後、データセットを１つ作成し、ノートブックで取得したデータを手元のPCにコピーして下さい。

```
scp -P 26500 kachaka@<カチャカのIPアドレス>:kachaka-api/python/demos/data/* coco-annotator/datasets/<作成したデータセットの名前>/
```

* アノテーションツールの使用方法に従い、バウンディングボックスのアノテーションを行って下さい。
* アノテーションが終了したらアノテーションファイルをエクスポートして下さい。
    * .exports以下にjsonが生成され、適切にエクスポートできていると内部に"annotations"フィールドが含まれています。

### 学習実行

* 以下のコマンドを実行すると転移学習が走ります。

```
./train.py --dataset <作成したデータセットへのパス> --epoch 100
```

### 評価

* 以下のコマンドを実行すると学習済モデルで推論が実行され、outdirで指定したディレクトリに結果画像が出力されます。

```
./eval.py --checkpoint latest_checkpoint.pt --indir <作成したデータセットへのパス> --outdir <出力ディレクトリ名>
```

### 学習済モデルをカチャカで実行

* 以下のコマンドでonnxファイルを出力します。

```
./create_onnx.py --checkpoint latest_checkpoint.pt --out predictor.onnx
```

* 生成されたonnxファイルをカチャカ体内に転送します。

```
scp -P 26500 predictor.onnx kachaka@<カチャカのIPアドレス>:kachaka-api/python/demos/
```

* カチャカ体内での推論実行方法は [ノートブック](https://github.com/pf-robotics/kachaka-api/blob/master/python/demos/run_custom_object_detection.ipynb) をご覧下さい。
