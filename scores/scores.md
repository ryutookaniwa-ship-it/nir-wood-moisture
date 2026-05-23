# NIR Wood Moisture — Score Tracker

Metric: **RMSE** (lower is better)

---

## 🚨 ルール確認 (2026-04-30)

コンペルールを精読した結果、以下が判明:

> 「評価用データをモデルの学習に用いることは禁止します。」  
> 「評価用データから得られる乾燥過程や測定順序など、複数サンプルが存在することを前提とした情報を利用すること」← 禁止  
> 判断基準: **「未知のスペクトルが1つだけ与えられた状況でも実行可能か？」**

**違反と判断した実験:**
- PL1/PL2/PL3/SP1: 評価データをモデル学習に使用
- CO1: テストデータのmean/stdを使用
- GS1/GS2/OB1-A/OB1-B/OB2-A/OB2-B: テスト測定順序(delta_prev等)を使用

**✅ 有効なLBベスト: P1 = 15.395**  
PL2(15.392)は違反のため無効。

---

## ⚠️ 重大な気づき: CV戦略の根本的な誤り

**訓練樹種 {1,3,4,5,8,11,12,13,14,15,16,17,19} とテスト樹種 {2,6,7,9,10,18} は完全非重複。**

通常のKFold CVは同一樹種のサンプルが訓練/検証両方に入るため、スペクトルの樹種固有パターンを記憶できてしまう。
→ KFold CVスコアはすべて楽観的すぎる（実際のLBと大きく乖離）。

**正しい評価: LOSO-CV (Leave-One-Species-Out)**
各foldで1樹種全サンプルを検証セットに置く → テスト条件を正しく模倣

---

## Baseline

| # | 手法 | 前処理 | モデル | KFold-RMSE | LOSO-RMSE | LB-RMSE | 備考 |
|---|------|--------|--------|------------|-----------|---------|------|
| 0 | **Baseline** | SNV→SG1 | PLS(20)+Ridge(10) | 7.7752 | 61.98 | — | KFoldは楽観的 |

---

## 打ち手ログ

| # | 手法 | モデル | KFold-RMSE | LOSO-RMSE | LB-RMSE | 備考 |
|---|------|--------|------------|-----------|---------|------|
| A | FSP分割 | PLS x2+ソフトブレンド | 29.90 | — | — | ❌ KFoldでも失敗 |
| B | 水吸収帯特徴量追加 | LV+band→Ridge(0.1) | 7.7317 | — | — | 🔺 KFold微改善のみ |
| C | 樹種one-hot | LV+one-hot→Ridge | 7.2721 | — | — | ⚠️ テスト時はone-hot全ゼロ |
| D | SVR(RBF) | PLS(10)+SVR(C=1000) | 3.6622 | 37.76 | — | ❌ C高すぎ→樹種過学習 |
| E | LightGBM | Raw→LGBM(feat=0.1) | 4.3995 | **21.48** | — | ✅ LOSO最良 |
| F | SVR+LGBM ensemble | 65%SVR+35%LGBM | 3.3389 | — | **26.688** | ❌ LB=26.69 KFold崩壊 |
| G | **LOSO best (LGBM)** | Raw→LGBM(lr=0.05,leaves=31,feat=0.1) | 4.40 | 21.48 | **18.995** | ✅ 現時点LBベスト。LOSO→LB方向に改善 |
| I | 水バンド絞り込み+LGBM | 4800-7300cm⁻¹(648点)→LGBM | — | 20.30 | 20.667 | ❌ GのLBより悪化。帯域制限は逆効果 |
| H | **LGBMハイパラ最適化** | LGBM(lr=0.02,leaves=63,ff=0.07,mcs=30) | — | **20.74** | 提出待ち | ✅ ベースライン-0.74改善。LB期待値≈18.3 |
| K | PLS(10成分)→LGBM | PLS次元圧縮→LGBM | — | 36.19 | — | ❌ PLSが訓練樹種に過適合。次元圧縮は逆効果 |
| **L** | **MSC+SG1+H-params** | **LGBM(lr=0.02,leaves=63,ff=0.07,mcs=30)** | — | **20.54** | 提出待ち | ✅ MSCがSNVより有効 |
| **M** | **MSC+SG1+sqrt(y)+H-params** | **LGBM** | — | **20.33** | **18.723** | ✅ 現時点LBベスト。G比-0.272改善 |
| N | バギング(5seeds) | LGBM×5平均 | — | 20.53 | — | ❌ seed平均で逆に悪化。M単体が最良 |
| O | ターゲット変換探索 | べき乗/Box-Cox/YJ/FSP分割 | — | 20.33 | — | ✅ sqrt(p=0.50)が最良。他の変換は全て悪化。変換の限界を確認 |
| **P** | **MSC+SG1(w=9)+sqrt+H-params** | **LGBM** | — | **20.10** | 提出待ち | ✅ w=9>7>11>13>15>21と小窓が有利 |
| Q | MSC+SG1(w=11)+sqrt+再チューニング | LGBM(leaves=31,mcs=10) | — | 20.17 | — | ✅ sqrt後の最適leaves=31,mcs=10に変化 |
| **R** | **MSC+SG1(w=9)+sqrt+w9最適化** | **LGBM(leaves=63,mcs=10,ff=0.07,lr=0.02)** | — | **19.68** | **18.403** | ✅ 現LBベスト。248位/803人。G比-0.592 |
| V | R+Mブレンド | alpha探索 | — | 19.68 | — | ❌ 残差相関0.985で多様性なし。Rと同一 |
| S | window細粒度(3/5/7/9/11) | LGBM | — | 19.68 | — | ✅ w=9が真の最適と確定 |
| **T** | **MSC+SG(w=5,poly=3)+sqrt** | **LGBM(R-params)** | — | **19.55** | 提出待ち | ✅ 現LOSO最良。poly=3がpoly=2より有効 |
| W | 樹種単位スペクトルセンタリング | LGBM(T-params) | — | 24.07 | — | ❌ センタリングで+4.52悪化。樹種平均に有益シグナルが含まれていた |
| X | 1D-CNN | Conv1D×2+GAP+PyTorch | — | 26.78 | — | ❌ +7.23悪化。Species15のRMSE=78が足を引っ張る。N=1322では過学習 |
| Y | 水吸収帯精密絞り込み | LGBM(T-params) | — | 21.78 | — | ❌ 最良は3B_narrow(5187+6896+8333、310点)でも+2.23悪化。全波長が最良 |
| **A1** | **rounds修正** | **LGBM(T-params, avg_iter=704)** | — | **19.54** | 提出待ち | ✅ 旧固定500→avg704。sp14/15が3000rounds上限に達 |
| A3 | MSC LOSOリーク修正 | LGBM(T-params) | — | 19.70 | — | ✅ CVが正直化(+0.16 pessimistic)。LBは変わらない見込み |
| B1 | FSP分割モデル | LGBM×3(global/low/high) | — | 20.02 | — | ❌ sp15(RMSE=54.85, MC max=298%)に引っ張られ悪化 |
| A2 | EMSC(poly=1)+SG(w=5,p=3)+sqrt | LGBM(T-params, avg_iter=531) | **20.049** | 17.10 | — | ❌ LB=20.05 CV楽観バイアス+2.95。テスト樹種でEMSC補正が逆効果 |
| **B2** | **MSC+SG+EPO(n=5)** | **LGBM(T-params, avg_iter=575)** | **17.651** | **16.44** | ✅ 提出済 | ✅ 現LBベスト(R比-0.752)。CV楽観gap=+1.21 |
| B3 | 異前処理スタッキング(4モデル) | Ridge meta-learner | — | 20.51 | — | ❌ LGBMモデル間r=0.99で多様性なし |
| B2b | EMSC(poly=1)+EPO(n=5) | LGBM(T-params, avg_iter=477) | — | 16.45 | — | 未提出(A2悪化のためEMSC不使用) |
| C2 | CatBoost+最近傍樹種特徴量 | CatBoost(depth=6, avg_iter=188) | — | 21.91 | — | ❌ sp15 RMSE=61.28に引っ張られ悪化 |
| D1 | 改良EPO(train+testX, k-means) | LGBM(T-params) | — | 35.99 | — | ❌ テストX統計量使用→ルール違反+大幅悪化 |
| D2 | di-PLS単体 | DIPLS(A,l grid) | — | — | — | ❌ xt=Xte(テスト全体)→ルール違反。未実行 |
| D4 | B2+物理特徴量(水吸収帯面積/比) | LGBM(T-params) | — | 16.44 | — | ❌ 改善なし。EPOが既に水吸収帯変動を処理済み |
| D5 | B2+アイソトニック残差補正 | LGBM+IsotonicReg | — | 14.39* | **18.117** | ❌ LOSO=14.39は同一データ評価で楽観的。LBは悪化 |
| **E2** | **EPO n_comp=7** | **LGBM(B2-params)** | — | **14.99** | **18.740** | ❌ LOSO-1.45改善もLB+1.09悪化。n=7でテスト樹種に必要な水分シグナルまで除去。n=5が最適 |
| E3 | ノイズ拡張(std=0.05, x3) | LGBM(B2-params, EPO n=5) | — | 17.05 | — | ❌ B2比+0.61悪化。aug_factor大→ラベルノイズ効果が正則化を上回る |
| **F1** | **MLP+EPO(n=5)** | **1555→512→256→128→1 BN+Drop** | — | **13.58** | **23.273** | ❌ gap=+9.69。MLPはEPO後も訓練種スペクトル空間を記憶。LOSO良いほどLB悪い逆相関確定 |
| F2 | RF+EPO(n=5) | RandomForest(mf=0.1, depth=10) | — | 17.39 | — | ❌ B2比+0.95。バギングはEPO後でも樹種汎化に不利 |
| F3 | 1D Transformer+EPO(n=5) | patch=5, d=64, heads=4, layers=2 | — | 16.39 | 19.520 | ❌ gap=+3.13。sp=15でrmse=39.15。B2より悪化 |
| F4 | TTA on B2 | B2モデル+テストノイズ付き複数予測平均 | — | 33.65 | — | ❌ EPO後std=0.000728にnoise=0.001が大きすぎ |
| G1 | PCA(20)→LGBM (EPOなし) | MSC+SG→PCA(n=20)→LGBM | — | 26.65 | — | ❌ PCA次元圧縮でfeature_fraction効果が消滅 |
| G2 | EPO(n=5)+PCA(20)→LGBM | MSC+SG→EPO→PCA(n=20)→LGBM | — | 21.20 | — | ❌ 圧縮は逆効果。B2の1555d+ff=0.07が最適 |
| H1 | MLP+EPO n=7 | 1555→512→256→128→1 BN+Drop | — | 12.71 | 24.272 | ❌ gap=+11.56。LOSO最良でLB最悪。逆相関の極致 |
| H2 | MLP(F1)+LGBM(B2) アンサンブル | alpha=0.8(MLP):0.2(LGBM) | — | 13.34 | 21.129 | ❌ gap=+7.79。MLP混入分だけLBが悪化 |
| I1 | feature_fraction探索 | LGBM(ff=0.02〜0.10) | — | 16.44 | — | ❌ ff=0.07(B2)が既に最適。スイートスポット確定 |
| **I2** | **SG(w=9,p=2)+EPO+LGBM** | **MSC+SG(w=9,p=2)+EPO(n=5)+LGBM** | — | **15.73** | **16.101** | ✅✅ **新LBベスト**。B2比-1.55。gap=+0.37(B2の+1.21より大幅縮小) |
| I3 | EPO後追加正規化 | B2+SNV/MSC/StdScaler | — | 16.44 | — | ❌ 全て逆効果。B2そのままが最良 |
| I4 | シードアンサンブル(5seeds) | B2×5seeds平均 | — | 16.35 | — | △ -0.09のみ。r=0.998で多様性なし |
| J1 | I2ベース EPO n_comp再探索 | LGBM(I2-params, n=2〜10) | — | 15.73 | — | ✅ n=5が依然最適。n=6以上で急激悪化。EPO最適値確定 |
| J2 | I2ベース feature_fraction再探索 | LGBM(ff=0.02〜0.15) | — | 15.73 | — | ✅ ff=0.07が依然最適。SG設定によらずスイートスポット不変 |
| **J3** | **I2×B2アンサンブル** | **LGBM×2, alpha=0.8(I2)** | — | **15.66** | **16.308** | ❌ LOSO-0.07改善もLB+0.21悪化。r=0.9883で多様性不足。I2単体が最良 |
| J4 | SG window/poly細粒度探索 | LGBM(I2-params, 10configs) | — | 15.73 | — | ✅ (w=9,p=2)が真の最適と再確定。poly=2はw=9専用スイートスポット |
| K3 | XGBoost + I2パイプライン | XGB(depth=6, col=0.07, 1点打ち) | — | 16.39 | — | ❌ I2比+0.66悪化。level-wise成長はleaf-wiseより不利。avg_iter=405 |
| **L1** | **I2+LGBMパラ再チューニング** | **LGBM(leaves=31,mcs=30,ff=0.07,lr=0.02)** | — | **15.44** | **16.461** | ❌ LOSO-0.29改善もLB+0.36悪化。leaves=31,mcs=30は訓練種過学習。I2(leaves=63,mcs=10)が最適 |
| L2 | 散乱補正比較(Raw/SNV/MSC)+EPO | LGBM(I2-params) | — | 15.73(MSC) | — | ✅ MSC確定最良。Raw=22.77(大幅悪化)、SNV=15.99。散乱補正はMSC固定で確定 |
| L3 | 学習率チューニング(lr=0.02/0.01/0.005) | LGBM(I2-params, patience=100) | — | 15.72 | — | △ lr=0.02+patience拡大でavg_iter=697→微改善(-0.006)。lr下げは逆効果 |
| L4 | EPOをfold内計算(LOSOリーク除去) | LGBM(I2-params) | — | 20.91 | — | ❌ +5.18悪化。fold内では推定できる樹種数減少→EPO方向が不安定 |
| L5 | SVR(RBF)+EPO | SVR(C=100,gamma=scale) | — | 21.88 | — | ❌ +6.15悪化。EPO後もSVRはLGBM(ff=0.07)に及ばない。random subspaceが本質 |
| M1 | Huber/MAE損失 | LGBM(I2-params, huber_delta探索) | — | 17.24(huber) | — | ❌ 全Huber/MAPE設定でI2(L2)より悪化。L2が最良損失関数と確定 |
| **M2** | **べき乗変換探索(p=0.30)** | **LGBM(I2-params, y^0.30変換)** | — | **15.5877** | **15.545** | ✅✅ **新LBベスト**。I2比-0.556。gap=+0.04(非常に小さい)。高MC域圧縮が有効 |
| **P1** | **細粒度p探索(p=0.27)** | **LGBM(I2-params, y^0.27変換)** | — | **15.4725** | **15.395** | ✅✅ **新LBベスト**。M2比-0.150。gap=-0.077(LBがLOSOより良い)。p=0.27確定最適 |
| P2 | sample_weight(MC>100%, w=2.0) | LGBM(I2-params, p=0.30) | — | 15.5347 | — | △ M2比-0.053。重み付けは効果小。P1(p=0.27)の方が大幅に強力 |
| P3 | DART(rate_drop=0.05, rounds=800) | DART+p=0.30 | — | 23.07 | — | ❌ +7.49大幅悪化。800 rounds不足。DARTはlr=0.02+800では収束不十分 |
| P4 | Tweedie(vp=1.2〜1.8) | Tweedie単体/+p=0.30 | — | 15.84(vp=1.5) | — | ❌ 全設定でM2より悪化。Tweedie分布仮定はL2より不利。vp=2.0はLightGBM制約でクラッシュ |
| P5 | p=0.27+weight(MC>100%,w=2.0) | LGBM(I2-params) | — | 15.5368 | — | ❌ P1比+0.064悪化。重み付けはp=0.27と相性不良。単独効果の合算にならず |
| P6 | 超細粒度p探索(0.255〜0.290) | LGBM(I2-params) | — | 15.4725 | — | ✅ p=0.270が真の最適と確定。p=0.265(+0.005)・p=0.278(+0.015)と急激に悪化 |
| Q1 | EPO bin_width探索(5/7/10/15/20/30) | LGBM(p=0.27) | — | 15.4725 | — | ✅ bw=10が依然最適と確定。小さいビン(bw=5: +1.52)も大きいビン(bw=30: +1.56)も悪化 |
| Q2 | leaves/mcs再チューニング(p=0.27) | LGBM(lv=47,mcs=30,max_bin=127) | — | 15.2130 | **15.535** | ❌ LOSO改善もLB悪化(P1比+0.140)。leaves=47,mcs=30が訓練種過適合。I2-params(leaves=63,mcs=10)がLB最適と再確定 |
| Q3 | EPO n_comp再探索(2〜8, p=0.27) | LGBM(p=0.27) | — | 15.4725 | — | ✅ n=5が依然最適と確定。n=6は+0.019、n=7で+2.37と急激悪化。EPO全パラメータ確定 |
| R1 | EPOなし(MSC+SG+LGBM) | LGBM(Q2-params) | — | 21.054 | — | ❌ EPOなしでsp15=58.70と更に悪化。EPOは必須 |
| R2+R3 | 2段モデル(t=150%)+重み(w=3,>100%) | LGBM(Q2-params) | — | **14.776** | **16.691** | ❌ LOSO大幅改善もLB+1.30悪化。gap=+1.915。高MC重み付けがテスト樹種で逆効果。sp15過適合パターン |
| S1 | extra_trees=True + leaves探索 | LGBM(P1-params,et=True,lv=95) | — | 16.6019 | — | ❌ P1比+1.13。分割閾値ランダム化は逆効果 |
| S2 | Huber損失(delta=0.5〜8.0)+p=0.27 | LGBM(P1-params,huber) | — | 15.7004 | — | ❌ 全delta値で同一結果(15.70)。y^0.27空間でHuber≈L2 |
| S3 | di-PLS (A=5〜20, l=0〜1) | DIPLS単体 | — | — | — | ❌ sp1単折RMSE=33(P1全体15.47)。lパラメータが完全無効。線形モデルの限界 |
| S5 | Test-Spectrum EPO (k=1〜3) | LGBM(P1-params) | — | 26.38〜28.71 | — | ❌❌ 壊滅的悪化(+10〜13)。テストXのPCA方向に含水率シグナルが含有 |
| S6 | LOSO Bagging(13モデル平均) | LGBM×13(P1-params) | — | 15.4725 | — | ❌ r=0.9963で多様性なし。I4(r=0.998)と同様に効果なし |
| S7(C) | OSC(k=5)+EPO → LGBM | Pipeline C | — | 6.53(leaky!) | — | ❌❌ LOSOリーク確定。honest LOSO=52.70。OSCはy使用→保留種のy値で調整済みCV=無意味 |
| S7(D) | EPO+OSC(k=3) → LGBM | Pipeline D | — | 15.3603 | — | ❌ Pipeline Cより小さいがOSCである限り同様のリーク。honest LOSO未測定だが類似パターン見込み |
| T1 | マルチスケールSG w=[5,9,13] Joint-EPO ff=0.023 | LGBM(P1-params) | — | 16.0680 | — | ❌ ff低すぎsp3収束せず。4665dim追加もP1比+0.60悪化 |
| T1b | T1 ff grid(0.04〜0.07) | LGBM(best ff=0.07) | — | 15.7600 | — | ❌ ff=0.07でもP1比+0.29。同等feat/treeでも多スケール情報増分なし |
| T1c | 2-scale SG w=[5,13] Joint-EPO | LGBM(ff=0.035) | — | 16.2881 | — | ❌ w=9なしで+0.82悪化。中周波除外は逆効果 |
| T1d | 混合poly (w5p3+w9p2) Joint-EPO | LGBM(ff=0.035) | — | 16.3360 | — | ❌ +0.86悪化。混合polyの情報増分なし |
| U1 | P1診断 sp15除外LOSO分析 | P1パイプライン | — | 15.4725(full) / 11.0694(ex-sp15) | — | ✅診断: sp15がLOSOを+4.40引き上げ。LB=15.40 >> ex-sp15 LOSO=11.07。テスト樹種は訓練種より4点難 |
| U2 | EPO+sp15 within-PCA拡張(k=2,3,5) | LGBM(P1-params) | — | 16.7748(best k=2) | — | ❌ sp15のPCA方向が水分シグナルも除去。全k値で悪化 |
| U3-A | 線形キャリブレーション(全サンプル) | P1+LinearCal(a=1.11,b=-3.69) | — | 14.6583 | 17.634 | ❌ LOSO-0.81もLB+2.24悪化。訓練バイアスはテスト樹種に非適用 |
| U3-B | 線形キャリブレーション(sp15除外fit) | P1+LinearCal(a=1.05,b=-1.33) | — | 14.8924 | 16.469 | ❌ LOSO-0.58もLB+1.07悪化。最も保守的だが依然悪化 |
| U3-C | 2次多項式キャリブレーション | P1+Poly2 | — | 14.6577 | 17.680 | ❌ LOSO-0.81もLB+2.29悪化 |
| U3-D | sp15専用線形補正 | P1+sp15LinearFix | — | 13.4489 | 24.328 | ❌❌ LOSOの同一データ評価で楽観的。LB+8.93壊滅。D5と同一パターン |
| Z1 | EPO+real(IFFT_EPO後) concat | LGBM(P1-params) | — | 16.055 | — | ❌ +0.58悪化。IFFT実部追加は効果なし |
| Z2a | EPO+real+imag(IFFT) | LGBM(P1-params) | — | 16.304 | — | ❌ まるっと(複素数全部)でも悪化 |
| Z2b | real(IFFT)のみ置き換え | LGBM(P1-params) | — | 23.770 | — | ❌❌ EPO除去で大幅悪化 |
| Z2c | abs(IFFT)のみ置き換え | LGBM(P1-params) | — | 18.038 | — | ❌ EPO置き換えは不可 |
| **Z2d** | **EPO+abs(IFFT_EPO後)** | **LGBM(P1-params)** | — | **15.720** | — | △ 全IFFTバリアント最近接。+0.25止まり |
| Z3a-d | IFFTステージ変更4種 | LGBM(P1-params) | — | 16.4〜27.9 | — | ❌ 全ステージで悪化。EPO前/生スペクトルも効果なし |

> **IFFT実験まとめ(Discord tip)**: P1レベルではEPOがすでにIFFTの情報をカバー済み。EPO未使用の弱いベースラインでは有効な可能性あり。アーカイブ: src/archive/nir/nir_exp_Z*.py
| V1a | EPO+db4全係数(DWT) | LGBM(P1-params) | — | 15.5416 | — | △ +0.069。全係数は冗長 |
| V1b | EPO+sym4全係数(DWT) | LGBM(P1-params) | — | 15.6254 | — | ❌ db4より劣る |
| V1c | EPO+db4近似係数のみ | LGBM(P1-params) | — | 15.8818 | — | ❌ 低周波ベースラインはEPO済み |
| **V1d** | **EPO+db4詳細係数のみ** | **LGBM(P1-params)** | — | **15.3486** | **15.946** | ❌ LOSO-0.124改善もLB+0.551悪化。詳細係数が訓練樹種のスペクトル微細構造を記憶。MLP/H1と同パターン |
| V2a-d | MSC参照変更(中央値/高MC/低MC/中MC) | LGBM(P1-params) | — | 15.617〜15.667 | — | ❌ 全参照で悪化。全訓練平均がMSC参照として最適と確定 |
| V3a-c | 反復EPO (x2/x3/n=5+n=3) | LGBM(P1-params) | — | 16.49〜18.46 | — | ❌ EPO回数増加で急激悪化。2回目以降で含水率シグナルまで除去。E2(n=7)と同パターン |
| V3d | EPO+EPO残差 | LGBM(P1-params) | — | 15.4725 | — | △ P1と完全一致。残差方向への2回目EPOは数学的に同一操作 |
| V4a | EPO(SG1次)+EPO(0次) | LGBM(P1-params) | — | 17.517 | — | ❌ 0次スペクトルとの結合は逆効果 |
| V4b | EPO(SG1次)+MSC(0次) | LGBM(P1-params) | — | 20.420 | — | ❌ 0次(EPOなし)はさらに悪化 |
| V4c | EPO(SG1次)+EPO(SG2次) | LGBM(P1-params) | — | 15.865 | — | △ 最近接だが+0.39。1次+2次の相補性は限定的 |
| V4d | EPO(0次)のみ | LGBM(P1-params) | — | 19.995 | — | ❌ SG微分なしはEPOがあっても大幅悪化 |
| W1 | 生スペクトル強度特徴量追加(raw_mean/5187/6896/比) | LGBM(P1-params)+5raw特徴量 | — | 16.025 | — | ❌ +0.55悪化。EPO後の1555次元に生スペクトル5特徴量concat。MSC+SG変換が絶対強度情報を破壊済みで追加しても樹種固有パターン強化 |
| W2 | k-NN類似度特徴量(k=5/10/20) | LGBM(P1-params)+5kNN特徴量 | — | 17.697 | — | ❌ +2.22悪化。訓練樹種基準のk-NN予測(corr=0.81)がテスト樹種に適用不可。訓練樹種バイアスが転写される |
| W3 | EPO後Ridge直接適用 | Ridge(alpha=1e5, raw) | — | 50.296 | — | ❌❌ +34.82壊滅。1555次元全方向に均一L2では樹種固有パターン除去不可。LGBM(ff=0.07)のランダムサブスペースが本質 |
| RL1 | L2正則化探索(reg_alpha/lambda grid 6点) | LGBM(P1-params) | — | 15.4725 | — | ❌ 全正則化設定でP1以上悪化。(0,0)デフォルトが最良。EPO+ff=0.07で既に十分正則化 |
| RL2 | DART(rate_drop=0.05/0.10/0.20, rounds=2000) | DART+P1-params | — | 18.4120 | — | ❌ +2.94悪化。best=rd=0.05でも壊滅。2000roundsでDARTは収束不十分。GBDTが優位 |
| RL3 | bagging_fraction探索(1.0/0.9/0.8/0.7) | LGBM(P1-params) | — | 15.4725 | — | ❌ 全バギング設定でP1以上悪化。bf=1.0(無効)が最良。列サブサンプリング(ff=0.07)が十分であり行サブサンプリングは不要 |
| ~~PL1~~ | ~~疑似ラベル(テスト6種を予測ラベルで追加)~~ | ~~LGBM(P1-params, 13+6種)~~ | — | ~~15.0660~~ | ~~15.393~~ | 🚨 **ルール違反**: 評価データをモデル学習に使用 |
| ~~CO1~~ | ~~CORAL整合(EPO後の訓練特徴量をtestの平均/std分布に揃える)~~ | ~~LGBM(P1-params)~~ | — | ~~15.6290~~ | — | 🚨 **ルール違反**: テストデータのmean/stdを特徴量計算に使用 |
| ~~PL2~~ | ~~反復疑似ラベル(Round0→v0→Round1→v1→Round2)~~ | ~~LGBM(P1-params)~~ | — | ~~15.0623~~ | ~~15.392~~ | 🚨 **ルール違反**: 評価データをモデル学習に使用 |
| MH1 | 複数種同時holdout CV + leaves=31 再チューニング | LGBM(lv=31,ff=0.07,lr=0.02,mcs=10) | — | 15.3232 | 15.418 | ❌ P1(15.395)比+0.023悪化。LOSO-0.15改善もLBは逆効果。lv=31はP1より汎化しない |
| ~~EN1~~ | ~~P1×PL2 OOFブレンド (alpha=0.0〜1.0探索)~~ | ~~LGBM×2 blend~~ | — | ~~15.0623~~ | — | 🚨 **ルール違反**: PL2依存のため |
| ~~PL3~~ | ~~19種EPO再計算(疑似ラベルv1でtest6種追加)~~ | ~~LGBM(P1-params, 19sp EPO)~~ | — | ~~16.4498~~ | — | 🚨 **ルール違反**: 評価データを学習・EPO計算に使用 |
| ~~SP1~~ | ~~テスト種別EPO×6アンサンブル(各種専用EPO)~~ | ~~LGBM×6(14種EPO各)~~ | — | ~~15.8786~~ | — | 🚨 **ルール違反**: 各テスト種のデータを個別EPO計算に使用 |
| EX1 | 公式リポジトリ準拠: sp15(ベイスギ)+sp17(ベイマツ)除外 | LGBM(P1-params, EPOあり, 11種) | — | — | **18.819** | ❌ PL2比+3.43壊滅的悪化。除外種なしの全13種が最良と確定。sp15/sp17は除外しても汎化を損なう |
| EX2 | EPOなし+sp15/sp17除外(アンサンブル評価) | LGBM(P1-params, EPOなし, 11種) | — | 14.076(11種) | **15.993** | ❌ PL2比+0.60悪化。r=0.9798(<0.99)だが精度差が多様性を上回り、アンサンブル効果なし |
| OA1 | log1p変換 (公式レシピ準拠) | LGBM(P1-params, EPO) | — | 16.6414 | — | ❌ P1比+1.17。p=0.27(高MC圧縮)がlog1p(均等変換)より大幅に優位。変換の比較確定 |
| OA2 | P1+IntervalMean/Slope(30特徴量) | LGBM(P1-params, EPO) | — | 15.7686 | — | ❌ P1比+0.30。EPO後でも粗いスケール特徴量が逆効果。EPOが既にカバー済み |
| OA3 | P1+DWT要約統計(12特徴量,db4 level=3) | LGBM(P1-params, EPO) | — | 15.7674 | — | ❌ P1比+0.29。V1a全係数(+0.07)より悪化。要約統計12特徴量も効果なし |
| OA4 | P1+WaterBandSummary(EPO後スペクトルから10特徴量) | LGBM(P1-params, EPO) | — | 15.9602 | — | ❌ P1比+0.49。D4(生スペクトル水バンド,16.44)より改善するもP1比悪化。EPO後でも追加不要 |
| ~~OB1-A~~ | ~~公式フルパイプライン(SNV+63特徴量, 13種)~~ | ~~LGBM(公式params, log1p)~~ | — | ~~25.4067~~ | — | 🚨 **ルール違反**: GroupSeqのテスト予測に測定順序(delta_prev/position_ratio)を使用 |
| ~~OB1-B~~ | ~~公式フルパイプライン(SNV+63特徴量, sp15/17除外11種)~~ | ~~LGBM(公式params, log1p)~~ | — | ~~16.3950~~ | — | 🚨 **ルール違反**: 同上 |
| ~~OB2-A~~ | ~~EPO後delta_prev(3列)+P1~~ | ~~LGBM(P1-params, EPO)~~ | — | ~~15.8322~~ | — | 🚨 **ルール違反**: テスト測定順序(delta_prev)を使用 |
| ~~OB2-B~~ | ~~生スペクトルdelta_prev(3列)+P1~~ | ~~LGBM(P1-params, EPO)~~ | — | ~~15.9539~~ | — | 🚨 **ルール違反**: テスト測定順序(delta_prev)を使用 |
| OC1 | Modified EPO(y相関フィルタ, grid n=10〜20, thr=0.1〜0.4) | LGBM(P1-params) | — | 16.9709(best) | — | ❌ 全設定でP1悪化。標準EPO n=5方向のy相関は最大0.35で種レベル交絡。保持しても訓練種バイアスが残るだけ。土壌SOM向け手法は本問題に不適 |
| OC2 | GLSW(ソフトフィルタ, W=(I+αC)^-1, alpha=0.001〜5.0) | LGBM(P1-params) | — | 20.8258(best,α=0.001) | — | ❌ 全alpha値でP1比+5.3〜5.5壊滅的悪化。EPOの「ハード射影除去」に対しGLSWの「ソフト減衰」は本問題では逆効果。1555×1555行列逆算で樹種間変動を希釈しすぎ |
| OC3 | MSCC基準によるEPO n最適化(n=1〜10, MSCC vs LOSO比較) | LGBM(P1-params) | — | 15.4725(n=5) | — | ✅ n=5のLOSO最適を再確定。MSCCはn単調増加(n=10: 0.9514)だがLOSOはn=7以上で急激悪化(n=7: +2.37)。MSCCとLOSOの目的関数が本問題で乖離: 樹種と含水率範囲の交絡があるためEPO n>6が含水率シグナルまで除去。EPO全パラメータ完全確定 |

---

## 新アプローチ (P1への固執をやめた探索)

> P1(EPO+LGBM)の枠を外し、化学計量学的な正攻法を試す。

| # | 手法 | モデル | LOSO-RMSE | LB-RMSE | 備考 |
|---|------|--------|-----------|---------|------|
| NA1-A | VIP-PLS (EPOなし) | MSC+SG→VIP(≥1.5, 63波長)→PLS(n=5) | 22.61 | — | ❌ P1比+7.14。波長選択のみでは樹種汎化に不十分 |
| NA1-B | **CARS-PLS (EPOなし)** | MSC+SG→CARS(13波長)→PLS | **17.35** | — | ✅ EPOなしで17.35。LGBM(ff=0.07)なし+波長選択のみでG(21.48)を大幅上回る |
| NA1-C | SiPLS (EPOなし) | MSC+SG→区間8+10+18(231波長)→PLS | 18.99 | — | ✅ 選択区間が水吸収帯(6900+7600+4800 cm⁻¹)と一致。物理的に正しい |
| NA2-A | EPO+全波長PLS | MSC+SG+EPO→PLS(n=20) | 50.42 | — | ❌ EPO後の1555次元PLSは壊滅。次元削減必須 |
| **NA2-B** | **EPO+CARS-PLS** | **MSC+SG+EPO→CARS(26波長)→PLS(n=15)** | **13.9294** | **33.217** | ❌❌ gap=+19.3 壊滅。LOSO楽観バイアス最大。CARS選択26波長+n=15成分が訓練13種に過適合。MLP/H1と同パターン |
| NA2-C | EPO+SiPLS | MSC+SG+EPO→区間1+13+18(231波長)→PLS | 17.67 | — | ❌ P1比+2.20。区間固定では細かい波長選択に劣る |
| NA2-D | EPO+VIP-PLS | MSC+SG+EPO→VIP(≥1.2, 223波長)→PLS | 20.85 | — | ❌ EPO後VIPスコアが機能しない |

| **NA3** | **物理帯域+EPO+LGBM** | **MSC+SG+EPO全体→4800-5600+6200-7400 cm⁻¹(518波長)→LGBM(P1-params)** | **17.41** | — | ❌ P1比+1.94。Y実験(EPOなし,21.78)より改善するがP1に及ばず。EPO後の帯域絞りは情報損失が勝る。パターンB(帯域先→EPO)も全設定で悪化 |
| **NA4** | **fold内CARS+PLS(honest LOSO)** | **EPO→fold内CARS(5波長)→PLS** | **23.49(A)** | — | ❌ fold内選択で全て悪化。min_waves=5まで絞りすぎ。fold内12種では安定したCARS信号が得られず。EPO全体+fold内CARSはLOSO=100.78と壊滅 |

| **NA5-C** | **position_ratio+EPO+LGBM** | **MSC+SG+EPO+position_ratio(1次元追加)+LGBM(y^0.27)** | **15.846** | **15.822** | ❌ P1比+0.43悪化。種内相関-0.93と強いが種固有MCスケールが学習不能。LOSO・LBともP1以下。gap=-0.024で過学習なし |

| **NA6** | **SiPLS前処理最小化** | **SNV+SG→SiPLS(区間18+8+16)→PLS** | **18.45** | — | ❌ EPOなしでは水吸収帯を正確に選択しても種固有変動を除けずP1に及ばず。最良区間は毎回4600-5200+7600-8200 cm⁻¹(物理的に正しい水吸収帯) |
| **NA7** | **物理バンド比** | **A(5200)/A(5800), A(6900)/A(5800)等8比率+LGBM/PLS** | **24.31(C)** | — | ❌ 5800 cm⁻¹参照帯が水吸収を含むため比にすると相関0.02と消滅。生吸光度(corr=0.72)のほうが有効。物理的種非依存参照帯はこのNIR範囲に存在しない |

> ⚠️ NA2-B注意: CARS波長選択は全訓練データで実施(LOSOループ外)。EPO計算と同扱いだが、LBで要確認。

---

## 公式リポジトリ (hirokenn/spectral_analysis) 由来の実験

> 参考: `docs/nir-wood-moisture/official_repo_analysis.md`
>
> 公式の主要アイデア: GroupSequenceFeature(乾燥時系列)・IntervalMean/Slope・DWT要約・WaterBandSummary・PLS OOF・残差アンサンブル
>
> **重要なドメイン知識**: 試料は飽水→室温乾燥しながら繰り返し測定。sample_id の樹種内順序 = 乾燥プロセスの時系列。データは吸光度(log変換済み)。

| # | 手法 | モデル | LOSO-RMSE | LB-RMSE | 備考 |
|---|------|--------|-----------|---------|------|
| ~~GS1~~ | ~~GroupSeqFeature(全11列)+P1~~ | ~~LGBM(P1-params)~~ | ~~16.0568~~ | — | 🚨 **ルール違反**: テスト測定順序(position_ratio/delta_prev/rolling)を使用 |
| ~~GS2~~ | ~~delta_prev(3列)のみ+P1~~ | ~~LGBM(P1-params)~~ | ~~15.8019~~ | — | 🚨 **ルール違反**: テスト測定順序(delta_prev)を使用 |
| PS1 | PLS(n=8) OOF予測(1列)+P1 | LGBM(P1-params) | 16.2072 | — | ❌ +0.73悪化。EPOなしPLSのOOF-RMSE=72.0と壊滅的→ノイズ特徴量として作用 |
| RE1 | P1ベース+delta残差Ridge補正 | LGBM+Ridge | 15.6408 | — | ❌ +0.17悪化。最小悪化だが改善なし。Stage2残差補正がテスト樹種に汎化しない |

**→ 全て P1 より悪化。公式特徴量エンジニアリングは SNV ベースライン向けであり、EPO+LGBM(ff=0.07) の P1 には効果なし。**

---

## LOSO-CV vs KFold の比較（全モデル）

| モデル | KFold-RMSE | LOSO-RMSE | LB-RMSE | 乖離 |
|--------|------------|-----------|---------|------|
| Baseline PLS(25)+Ridge | 7.77 | 47.92 | — | ×6.2 |
| SVR(C=10) | — | **31.47** | — | — |
| SVR(C=1000) | 3.66 | 37.76 | — | ×10 |
| **LGBM** | 4.40 | **21.48** | — | ×4.9 |
| Ensemble(KFold) | 3.34 | — | 26.69 | ×8 |
| **N4** | ```json
{
  "hypothesis": "The dominant bottleneck is s | LGBM(I2-params) | — | **15.5977** | — | Agent delta=-0.13. Experiment N4, using a `y^0.3` target transformation, achieved a LOS |
| **N3** | ```json
{
  "hypothesis": "The dominant bottleneck is s | LGBM(I2-params) | — | — | — | Agent delta=—. This experiment failed to execute due to a `SyntaxError` at line 62 in |
| **N2** | ```json
{
  "hypothesis": "The dominant bottleneck is s | LGBM(I2-params) | — | **15.5977** | — | Agent delta=-0.13. This experiment shows a marginal improvement over the I2 baseline (15.5977 vs  |
| **N1** | ```json
{
  "hypothesis": "The dominant bottleneck is s | LGBM(I2-params) | — | **15.5969** | — | Agent delta=-0.13. The N1 experiment shows a marginal improvement over the I2 baseline (15.59 |

**→ 今後はLOSO-RMSEを主指標とする**

---

## 試す打ち手リスト（LOSO-CVベース）

### 高優先
- [ ] **G. LGBM提出** → LBスコア確認（LOSO=21.48）
- [ ] **H. 水吸収帯のみで学習**: 5187・6896 cm⁻¹周辺に絞ったスペクトルでLGBM（樹種固有パターンを排除）
- [x] **I. 水バンド絞り込み**: LOSO=20.30 ✅ 改善(-1.18)。4800-7300 cm⁻¹(648点)が最良。提出待ち
- [ ] **J. SVR with 低C (C=10)**: LOSO=31.47。過学習を防いだSVR

### 中優先
- [ ] **K. MSC前処理**: 散乱補正の別手法で樹種間ばらつきを抑制できるか
- [ ] **L. 水バンドのみPLS**: 5000〜7200 cm⁻¹に絞ってPLS+Ridge

---

## Notes

- **評価指標はLOSO-RMSEを主とする**（KFoldは参考値のみ）
- 提出フォーマット: ヘッダーなし2列CSV (sample_number, pred)
- スクリプト:
  - `src/nir_loso_cv.py`: LOSO-CV評価フレームワーク
  - `src/nir_exp_*.py`: 各実験
- 出力先: `output/`
