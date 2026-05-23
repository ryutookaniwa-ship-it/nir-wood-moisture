# フレームワーク設計リファレンス

> 出典: hirokenn/spectral_analysis
> NIRコンペ用の構造化された実験フレームワーク。今後のリファクタリング時の参考。

## 設計思想

4つの責務を分離:
1. **データ表現** (`Dataset`) — X, y, groups, sample_id, wavenumbers
2. **条件定義** (`Recipe`) — preprocessor_name + model_name の参照ペア
3. **実行フレームワーク** (`Pipeline` / `Step` / `State`) — 順次実行
4. **CLI** — cv / submission のエントリポイント

## Recipe パターン

```python
@dataclass(frozen=True)
class Recipe:
    name: str
    preprocessor_name: str  # → preprocess_params.json の設定名
    model_name: str         # → params.json のモデル名
```

利点:
- CV と提出で同じ条件を再利用
- fold ごとにBuilderが新インスタンス生成 → リーク防止
- JSON定義と分離 → 前処理/モデルの差し替え容易

## 前処理アーキテクチャ

```
BasePreprocessor (Protocol)
  ├── fit(ds: Dataset) -> None
  ├── transform(ds: Dataset) -> Dataset
  └── fit_transform(ds: Dataset) -> Dataset

ComposablePreprocessor (>> で連結可能)
  ├── PreprocessingPipeline (直列合成)
  └── FeatureUnion (並列合成 → 横結合)
```

### 実装例: snv_lgbm パイプライン

```
snv_lgbm_pipeline
  → FeatureUnion
    → branch 1: SNV → spectral features
    → branch 2: GroupSequenceFeatures
```

### 既存 transforms
- IdentityPreprocessor (no-op)
- SNVPreprocessor (行ごとの平均0/std1正規化)
- SavitzkyGolayPreprocessor (scipy.signal.savgol_filter)

### 既存 feature extractors
- IntervalMeanFeatureExtractor (区間平均)
- IntervalSlopeFeatureExtractor (区間回帰傾き)
- DWTFeatureExtractor (離散ウェーブレット係数)
- WaterBandSummaryFeatureExtractor (水関連帯域の統計)
- GroupSequenceFeatureExtractor (グループ内順序の差分・移動統計)

## Pipeline パターン

```python
pipeline = MlflowStartRun() >> TrainCV() >> EvaluateOOF() >> PlotOOF() >> MlflowEndRun()
state = TrainState(dataset=train_ds)
result = pipeline.run(state)
```

### CV学習フロー
```
train.csv → Dataset → TrainState → TrainCV → EvaluateOOF → PlotOOF → metrics/plots
```

### 提出フロー
```
train + test → Dataset → TrainState → TrainFull → PredictTest → submission.csv
```

## JSON設定

### preprocess_params.json
- 単体: `{"build_type": "snv", "params": {}}`
- 直列: `{"build_type": "pipeline", "steps": [...]}`
- 並列: `{"build_type": "feature_union", "branches": [...]}`

### params.json
- モデルパラメータ定義

## 今後のリファクタリングで活用するポイント

1. **Recipe → JSON分離**: 現在の100+実験スクリプトを Recipe + JSON設定に移行可能
2. **FeatureUnion**: MSC + EPO + SG を branch 化して条件管理
3. **State パターン**: OOF予測、テスト予測、メトリクスを一元管理
4. **LOSOCV 組み込み**: フレームワーク内でLOSO-CVを標準化
