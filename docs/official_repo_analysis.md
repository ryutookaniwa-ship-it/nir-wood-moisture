# 公式リポジトリ (hirokenn/spectral_analysis) 解析メモ

## 1. リポジトリ概要

- URL: https://github.com/hirokenn/spectral_analysis
- ブランチ: 8つ (レシピ別に分岐している可能性)
- 主要ファイル: `src/`, `params.json`, `preprocess_params.json`

---

## 2. 重要なドメイン知識 (AGENTS.md より)

### 測定プロセス（最重要）

```
試料: 20mm × 20mm × 12mm の木材ブロック (19樹種)
密度: 0.32〜0.79 g/cm³

手順:
1. 水中に約1週間沈めて飽水状態にする
2. 室温環境で乾燥させながら、重量+NIRスペクトルを繰り返し測定
3. 重量変化がなくなるまで継続
```

**→ データの各行 = 乾燥中の1回の測定。sample_id の樹種内順序 = 時系列。**

乾燥速度が樹種ごとに異なるため、樹種ごとの測定数が異なる。

### 測定条件

- FT-NIR, プローブを試料表面から約 2mm の位置に設置
- 拡散光取得、NIR光の浸透深さは表面数mm程度
- スペクトル: 10000〜4000 cm⁻¹, 分解能 8 cm⁻¹, 32 scan 積算
- **吸光度 = リファレンス白色板と試料の反射光から計算**
- 板目・柾目・追い柾面で測定

### データ単位の確定

> 「各セルの値は、その波数でのスペクトル値（**吸光度**）」

→ データは既に吸光度。log(1/R) 変換は不要。

### Train/Test スペクトル分布の差

| | Train mean | Test mean | 差 |
|--|--|--|--|
| 全体 | 0.1330 | 0.0940 | -0.039 |
| 6996 cm⁻¹ | 0.2056 | 0.1639 | -0.042 |

テストスペクトルが系統的に低い。MSC の参照を訓練平均にすることで補正。

---

## 3. 公式パイプラインの全体像

### snv_lgbm レシピ (公式メイン)

```
入力スペクトル (1555 次元)
  │
  ├── SNV
  │     └── FeatureUnion:
  │           ├── IntervalMean(100点区間) → 15 特徴量
  │           ├── IntervalSlope(100点区間) → 15 特徴量
  │           ├── DWT(db4, level=3, mean/std/energy) → 12 特徴量
  │           └── WaterBandSummary(7000/5200, 5統計量) → 10 特徴量
  │                                          計 ~52 特徴量
  │
  └── GroupSequenceFeatures(5200/7000, window=5) → 11 特徴量
                                          計 ~63 特徴量
→ LGBM (colsample_bytree=0.8)
```

### savgol_lgbm レシピ

```
SG(window=11, poly=3) → 同じ FeatureUnion → LGBM
```

### 残差アンサンブル (base_pls_snv_lgbm_residual)

```
Stage1: identity → PLS(n=8) → 粗予測
Stage2: snv_lgbm_pipeline → LGBM → 残差予測
Final : Stage1 + Stage2
```

---

## 4. 各特徴量の実装詳細

### GroupSequenceFeatures (`group_sequence_features.py`)

```python
# sample_id 昇順で樹種内ソート後、以下を計算:
group_position_index   # 測定順位 (1, 2, 3, ...)
group_position_ratio   # 正規化順位 (0.0=最初, 1.0=最後)
delta_prev_5200        # 前回測定との 5200 cm⁻¹ 吸光度差分
delta_prev_7000        # 前回測定との 7000 cm⁻¹ 吸光度差分
delta_prev_global_mean # 前回測定との全波長平均差分
rolling_mean_5_5200    # 直近5測定の 5200 cm⁻¹ 移動平均
rolling_std_5_5200     # 直近5測定の 5200 cm⁻¹ 移動標準偏差
rolling_mean_5_7000    # 同上 (7000 cm⁻¹)
rolling_std_5_7000     # 同上
rolling_mean_5_global  # 同上 (全波長平均)
rolling_std_5_global   # 同上
```

**テスト時も使用可能** (test.csv に sample_number と species_number あり)

### IntervalMean/Slope (`interval_features.py`)

- 1555 列を 100 点ずつ 15 区間に分割
- 各区間の **平均値** と **線形傾き** を特徴量化
- 粗い周波数スケールでのスペクトル形状把握

### WaterBandSummary (`water_band_summary.py`)

- 7000 ± 200 cm⁻¹ 帯域 (6800〜7200)
- 5200 ± 200 cm⁻¹ 帯域 (5000〜5400)
- 各帯域で: mean / std / min / max / area (台形積分)
→ 計 10 特徴量

### DWTFeatures (`dwt_features.py`)

- Wavelet: db4, level=3, mode=symmetric
- 近似係数 (a3) + 詳細係数 (d3, d2, d1) の 4 サブバンド
- 各サブバンドで: mean / std / energy
→ 計 12 特徴量

### PLSOOFFeature (`pls_oof_feature.py`)

- LOSO-CV で PLS(n=8) の OOF 予測値 (スカラー1列) を生成
- 訓練: OOF 予測 / テスト: 全データ PLS 予測
- スタッキングの1段目メタ特徴量

---

## 5. 我々の実験との対応

| 公式特徴量 | 我々の実験 | 結果 |
|-----------|-----------|------|
| GroupSequence(全11列) | GS1 | LOSO +0.58 悪化 |
| delta_prev のみ(3列) | GS2 | LOSO +0.33 悪化 |
| PLS OOF(1列) | PS1 | LOSO +0.73 悪化 |
| P1+Ridge残差 | RE1 | LOSO +0.17 悪化 |
| WaterBandSummary相当 | D4, Y | 改善なし |
| DWT 係数全体 | V1a-d | LOSO改善→LB悪化 |

**全て P1 (LOSO=15.4725) より悪い。**

---

## 6. 未試行で P1 に追加可能なアイデア

| アイデア | 実装 | 期待値 | 備考 |
|---------|------|--------|------|
| log1p ターゲット変換 | `np.log1p(y)` | 低〜中 | p=0.27 と比較 |
| IntervalMean+Slope | 30 特徴量追加 | 低 | EPO後では冗長の可能性 |
| DWT 要約統計 (V1との違い) | 12 特徴量 | 低 | V1aが +0.07 で最近接だった |
| WaterBand 後 EPO (D4の亜種) | 10 特徴量 | 低 | D4が改善なしだった |

**結論: 公式リポジトリの特徴量エンジニアリングは彼らの SNV ベースラインには有効だが、EPO+LGBM(ff=0.07) の P1 には追加効果がない。EPO が公式の特徴圧縮と同等の役割を果たしている。**

---

## 7. 公式 LGBM パラメータ (参考)

```json
{
  "n_estimators": 500,
  "learning_rate": 0.03,
  "num_leaves": 31,
  "min_child_samples": 10,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.1,
  "reg_lambda": 0.1
}
```

我々の P1 との差異:
- `colsample_bytree=0.8` vs 我々の `feature_fraction=0.07` (大幅に異なる)
- 公式は 63 特徴量に対して 80% サンプリング ≒ 50 特徴量
- 我々は 1555 特徴量に対して 7% サンプリング ≒ 109 特徴量
- 実質的に1ツリーあたりの特徴数は近い

---

## 8. 公式 target transform

`recipes.py` に `log1p` 変換のレシピあり:
- `snv_lgbm_log1p`
- `savgol_lgbm_log1p`

公式は log1p を試しているが、我々は p=0.27 に最適化済み。log1p と p=0.27 の比較:

| MC(%) | log1p | y^0.27 |
|-------|-------|--------|
| 0 | 0.000 | 0.000 |
| 30 | 3.434 | 2.388 |
| 100 | 4.615 | 3.631 |
| 298 | 5.699 | 4.570 |

p=0.27 は低MC域を広げ高MC域を圧縮。log1p は中間域をより均等に扱う。

---

## 9. 現時点での知見サマリ

### P1 が圧倒する理由

公式: SNV → 特徴量圧縮(~50次元) → LGBM(colsample=0.8)
P1 : MSC → SG → EPO(n=5) → LGBM(ff=0.07)

EPO が「樹種固有の分散方向を射影除去」することで、公式の特徴量圧縮が解決しようとした問題をより直接的に解決している。

### P1 からの改善が難しい理由

- 追加特徴量は「訓練樹種固有のパターン」を含むためLB悪化
- モデル変更 (MLP/RF/SVR) もすべてLB悪化
- EPO の超パラメータ (n=5, bw=10) は徹底探索済み
- ターゲット変換 (p=0.27) も徹底探索済み

### 残る可能性

- **複数樹種同時 holdout CV** による適切なハイパラ探索 (精度改善より探索改善)
- **測定プロセス情報の別活用** (例: 飽水→乾燥の物理モデル)
