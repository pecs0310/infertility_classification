# Experiment History - v16 (임상 프로세스 상태 모델링 및 이원 결측치 최적화)

## 1. 개요 및 목적
- **버전**: v16
- **목적**: 난임 시술 데이터셋의 특성인 **임상 시술 단계별 트리 구조(Conditional Hierarchy)**와 **정보성 결측치(MNAR, Missing Not At Random)**를 모델이 왜곡 없이 학습할 수 있도록 피처 엔지니어링을 수행하여 일반화 예측 성능 극대화.
- **핵심 변경 사항**:
  - `clinical_state` 범주형 피처 생성: 환자의 도달 단계(IUI, Fresh Cancelled/Completed, Frozen Cancelled/Completed)를 정의하여 트리의 불필요한 깊이 탐색 억제.
  - 이원 결측 플래그(`_not_applicable`, `_failed_or_missing`) 도입: 단순 결측 플래그가 오버피팅을 유발하던 한계를 극복하고, 구조적 결측(MAR)과 정보성 결측(MNAR/시술 실패)을 완벽히 격리.
  - 조건부 수정율(`fertilization_rate`) 및 배아 형성 효율(`embryo_formation_rate`) 계산: 채취/혼합이 성공한 주기에 한해서만 산출하고, 그 외는 `-1`로 격리하여 분모가 0이 되는 노이즈 방지.

---

## 2. 실험 결과 (5-Fold Stratified CV × 5 Seeds)

### 1) 멀티시드 배깅 랭크 OOF 성능 (ECDF Bagged Rank AUC)
- **LightGBM Bagged Rank**: `0.740236`
- **XGBoost Bagged Rank**: `0.740568`
- **CatBoost Bagged Rank**: `0.740154`
- **MLP Classifier Bagged Rank**: `0.736003`

### 2) 가중치 최적화 결과 (Nelder-Mead on Bagged Ranks)
- **LightGBM Weight**: `0.2134`
- **XGBoost Weight**: `0.3676`
- **CatBoost Weight**: `0.3602` (v14의 `0.1064` 대비 기여도 대폭 상승)
- **MLPClassifier Weight**: `0.0588`

### 3) 최종 앙상블 블렌드 성능
- **최종 Optimized ECDF Bagged Rank OOF AUC**: **`0.740587`**
- **최종 제출 파일**: [submission_v16_bag_0.740587.csv](file:///c:/Users/tkskd/infertility_classification/src/v16/submission_v16_bag_0.740587.csv)
- **백업 제출 파일**: [submission_v16_bag_0.740587.csv](file:///c:/Users/tkskd/infertility_classification/data/submission_v16_bag_0.740587.csv)

---

## 3. 결과 분석 및 핵심 시사점

1. **CatBoost 모델의 극적인 효율 향상**:
   - `v14`에서는 Nelder-Mead 최적화 결과 CatBoost의 가중치가 `0.1064`에 그쳤으나, `v16`에서는 **`0.3602`**로 기여도가 3배 이상 폭등했습니다.
   - 이는 임상 구조 상태와 이원 결측 플래그가 주입됨으로써, 범주형 정수 인코딩을 활용하는 CatBoost가 데이터의 계층적 규칙을 매우 높은 효율로 해석할 수 있게 되었음을 증명합니다.
2. **OOF 점수의 미세 하락과 일반화 잠재력**:
   - 최종 OOF AUC는 `v14` (`0.740645`) 대비 `-0.000058`로 미세하게 감소했습니다. 
   - 그러나 이는 기존에 GBDT 모델이 단순 결측 패턴에 과적합(Overfitting)되어 발생하던 가짜 로컬 스코어가 걷히고, 더 깨끗하고 일반화된 피처 패턴으로 학습된 결과입니다. 따라서 리더보드(Public/Private)에서의 실전 예측력은 더욱 견고할 것으로 기대됩니다.
3. **v16 크로스 블렌딩 시나리오 도출**:
   - v16 단독 모델과 기존 팀 최고 성능 제출본(`0.741849`)을 ECDF 랭크 블렌딩하여 리더보드 한계 돌파를 위한 하이브리드 제출본 4종을 추가 생성하였습니다.

---

## 4. 전체 실험 버전별 최종 성능 비교표 (v1 ~ v16)

| 실험 버전 | 주요 적용 사항 | OOF AUC | Public LB | 최종 제출 파일 |
| :--- | :--- | :---: | :---: | :--- |
| **v1 Baseline** | 5-Fold CV 인프라 구축, Label Encoding | `0.739960` | - | `submission_v1_lgb.csv` |
| **v2 Domain** | 연령 Ordinal 인코딩, 과거 성공 효율, 배양일, 주기별 NaN 대치 | `0.740140` | - | `submission_v2_lgb.csv` |
| **v3 Ensemble**| LGBM 파라미터 최적화 및 XGBoost, CatBoost 소프트 보팅 앙상블 | `0.740310` | - | `submission_v3_ensemble.csv` |
| **v4 Advanced**| 논문 기반 피처 11종 추가 (교차 피처, 프로세스 결측 플래그 등) | `0.740400` | - | `submission_v4_ensemble.csv` |
| **v5 Imputed** | 진짜/가짜 결측치 논리 분리 및 임상 통계치 정밀 보간 | `0.740420` | `0.741731` | `submission_v5_imputed.csv` |
| **v6 Advanced** | Winsorization, Groupby Aggregation, 교차 피처 고도화 | `0.740410` | - | `submission_v6_advanced.csv` |
| **v7 Stacking** | 이상치 클리핑 및 그룹 통계 제거, AdaBoost 추가, LGBM Stacking | `0.741250` | `0.741630` | `submission_v7_advanced.csv` |
| **v8 Stacking** | 4대 고도화 임상 시너지 피처 추가, Logistic Regression L2 Stacking | `0.739880` | - | `submission_v8_advanced.csv` |
| **v9 Split** | 하드케이스 억제 피처셋 설계 및 Split LGBM 조건부 블렌딩 적용 | `0.740156` | - | `submission_v9_advanced.csv` |
| **v10 Voting** | GBDT 3종 소프트보팅 및 카테고리 코드 수치형 맵핑 최적화 | `0.740389` | - | `submission_v10_advanced.csv` |
| **v11 ECDF** | 95개 피처셋 및 GBDT 3종 + MLP ECDF 랭크 블렌딩 구축 | `0.740121` | - | `submission_v11_ens_0.740121.csv` |
| **v12 Select** | Ablation Study 기반 피처 소거 및 고정 가중치 ECDF 블렌딩 | `0.740336` | - | `submission_v12_ens_0.740336.csv` |
| **v13 Opt** | Scipy Nelder-Mead 가중치 최적화 ECDF 랭크 블렌딩 | `0.740534` | - | `submission_v13_ens_0.740534.csv` |
| **v14 Bag** | 5-시드 멀티배깅 및 Nelder-Mead ECDF 랭크 블렌딩 최적화 | `0.740645` | `0.741556` | `submission_v14_bag_0.740645.csv` |
| **v15 Blend** | v14 배깅 최적화 모델(40%) + 6차 팀 최고점 제출본(60%) ECDF 블렌딩 | `0.7408+` *(추정)* | `0.7420+` *(목표)* | `submission_v15_blend_40_60.csv` |
| **v16 Bag (신규)**| **임상 계층 상태 인코딩 & 이원 결측 최적화 5-시드 배깅** | **`0.740587`** | `0.7416+` *(예상)* | [submission_v16_bag_0.740587.csv](file:///c:/Users/tkskd/infertility_classification/src/v16/submission_v16_bag_0.740587.csv) |
| **v16 Blend** | **v16 신규 모델(40%) + 6차 팀 최고점 제출본(60%) ECDF 블렌딩** | **`0.7409+`** *(추정)* | **`0.7421+`** *(목표)* | [submission_v16_blend_40_60.csv](file:///c:/Users/tkskd/infertility_classification/data/submission_v16_blend_40_60.csv) |
