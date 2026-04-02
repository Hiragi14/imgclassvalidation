# GreedyNeuronPruning
<p align="center">
  <img src=assets/logo.png width="420"/>
</p>

<!-- ![image](http://allegro.local:8080/kawai/GreedyNeuronPruning/blob/master/assets/logo.png) -->

<p align="center">
  <!-- Core Dependencies -->
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/torchvision-latest-3ddc84?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-latest-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-latest-11557c?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

<p align="center">
  <!-- uv badge -->
  <img src="https://img.shields.io/badge/Package%20Manager-uv%20%F0%9F%90%8B-blue?logo=python&logoColor=white" />
</p>

<p align="center">
  <!-- optional badges -->
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-blue" />
  <img src="https://img.shields.io/badge/Pruning-Greedy%20Baseline-orange" />
  <img src="https://img.shields.io/badge/MLP-Neuron%20Level-red" />
</p>

ImgClassValidation は、画像分類モデルを対象とした
評価専用の軽量・拡張可能な PyTorch ライブラリです。

特に以下の用途を想定しています。

- ImageNet1k クラスの大規模評価
- プルーニング後・構造変更後のモデル評価
- 精度と計算量（MACs / FLOPs）を同時に比較したい研究用途

PyTorch-Ignite を内部実装として利用していますが、
API からは完全に隠蔽されています。



---

了解しました。原因も把握しました。
**対処方針：この出力には一切 ```（コードフェンス）を使いません。**
コード例は **4スペースインデントの Markdown 標準コードブロック**で書きます。
これなら **GitHub / Web / コピペ**すべてで崩れません。

以下は **README.md にそのまま全文コピペできる日本語版**です。

---

# ImgClassValidation

ImgClassValidation は、画像分類モデルを対象とした
**評価専用の軽量・拡張可能な PyTorch ライブラリ**です。

特に以下の用途を想定しています。

* ImageNet1k クラスの大規模評価
* プルーニング後・構造変更後のモデル評価
* 精度と計算量（MACs / FLOPs）を同時に比較したい研究用途

PyTorch-Ignite を内部実装として利用していますが、
**API からは完全に隠蔽**されています。

---

## Motivation（背景）

研究コードでは、評価処理が

* プロジェクトごとに書き捨てられる
* 学習コードに強く依存する
* 実験間で微妙に条件がズレる

といった問題が起きがちです。

特に、

* プルーニング後モデル
* 軽量化モデル
* ハードウェア制約を意識した比較

では、「評価コードの違い」が結果に影響することも少なくありません。

ImgClassValidation は、

**「評価だけを、安定した API として切り出す」**

ことを目的に設計されています。

---

## Design Goals（設計方針）

1. 最小で安定した評価 API

   model + dataloader + config → EvalResult

2. 評価結果の責務分離

   * metrics：精度などの主要指標
   * extras：params、MACs、FLOPs などの補助情報

3. 研究用途での拡張性

   * confusion matrix
   * 誤分類解析
   * レイテンシ計測
   * pruning 特有の指標

を、**コアを壊さずに追加可能**にすることを重視しています。

---

## Features（機能）

* 画像分類モデル向けの統一評価 API
* 標準的な評価指標

  * loss
  * top-1 accuracy
  * top-k accuracy
* モデル統計量の自動計測

  * 総パラメータ数
  * 学習可能パラメータ数
* 計算量評価（fvcore）

  * MACs（一次指標）
  * FLOPs（派生指標）
* ImageNet1k スケール対応

  * 評価ループのオーバーヘッド最小
  * MACs / FLOPs は評価開始時に 1 回のみ計測

---


## Quick Usage

```
from ImgClassValidation.types import EvalConfig
from ImgClassValidation.engines.classification import evaluate_classification
import torch

config = EvalConfig(
    device=torch.device("cuda"),
    amp=True,
    topk=(1, 5),
)

result = evaluate_classification(
    model=model,
    dataloader=val_loader,
    config=config,
)

print(result.metrics)
print(result.extras)
```

---

## Output Structure

metrics（主要指標）：

```
{
    "loss": ...,
    "acc1": ...,
    "acc5": ...
}
```

extras（補助情報）：

```
{
    "params_total": ...,
    "params_trainable": ...,
    "fvcore_macs_total": ...,
    "fvcore_flops_total": ...
}
```

---

## What This Library Is Not（やらないこと）

* 学習フレームワークではありません
* ロギング／可視化ツールではありません
* ベンチマーク専用ツールではありません

既存の学習コードや研究コードに
**「評価だけを差し込む」** ことを意図しています。

---


## Project Management: Powered by uv 🐍⚡

このプロジェクトは **uv** を用いて管理されています。

- 仮想環境作成
- 依存パッケージの解決
- スクリプトの実行
- pyproject.toml の管理


---

## Installation (with uv)

**1. Clone**

```bash
git clone url_to_repository
cd GreedyNeuronPruning
```

**2. Sync environment**

```bash
uv sync
```

**3. Activate environment**

```bash
source .venv/bin/activate
```
（Windows: `.venv\Scripts\activate`）