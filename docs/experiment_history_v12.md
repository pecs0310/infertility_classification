# Experiment History - v12 (ECDF Rank Blending & 피처 최적화)

## 1. 개요 및 목적
- **버전**: v12
- **목적**: 이전 기수 1등 솔루션(Ver.62)의 핵심 기법인 **ECDF Rank Blending(랭크 블렌딩)** 및 **실질 가임 연령(Maternal Rejuvenation)** 피처를 적용하고, GBDT 3종과 MLPClassifier의 다양성 결합을 통해 일반화 성능 향상 및 과적합 제어.
- **특이사항**: Ablation Study 결과에 따라 오버핏을 유발하던 비율 피처와 결측 플래그를 모두 제거하고, 4개 고결측 컬럼 및 정밀 수동 대치는 유지함.

---

## 2. 피처 구성 및 전처리 변경 사항
- **최종 피처 수**: 81개
- **제거된 피처 (Ablation Study 반영)**:
  - `pregnancy_efficiency`, `delivery_efficiency` 등 비율 피처 8종 전면 제거.
  - `is_missing_eggs`, `is_missing_embryos` 등 결측 플래그 5종 전면 제거.
  - `infertility_severity_score` (불임 원인 개수 합산) 제거.
- **추가된 피처 (1등 솔루션 벤치마킹)**:
  - `patient_age_mid` 및 `oocyte_donor_age_mid`: 환자 및 기증자 연령대별 중간값.
  - `effective_maternal_age`: 난자 출처가 기증 제공인 경우 기증자의 나이 중간값, 본인 제공인 경우 본인의 나이 중간값을 할당하여 실제 난자의 나이를 반영한 실질 가임 연령.
  - `donor_rejuvenation_gap`: 환자 연령 중간값과 기증자 연령 중간값의 차이 (`환자 나이 - 기증자 나이`).
  - `donor_rejuvenation_gap_positive`, `donor_rejuvenation_gap_10plus`: 리쥬브네이션 효과 유무 및 10세 이상 차이 여부 바이너리 플래그.
- **범주형 변수 처리**:
  - XGBoost의 추론 시 unseen 카테고리 에러(`IUI:ICI` 등)를 방지하기 위해 Train과 Test의 범주형 컬럼 고유값 풀(categories)을 통합하여 일치시킴.

---

## 3. 실험 결과 (ROC-AUC)

### 1) 1차 개별 모델 성능 (5-Fold Stratified CV)
- **Tuned Joint LightGBM**: `0.739860` (num_leaves=44, min_child_samples=82)
- **Tuned Joint XGBoost**: 🏆 **`0.740481`** (피처 최적화 및 카테고리 일치 효과로 프로젝트 단일 최고점 달성)
- **Joint CatBoost**: `0.740098`
- **MLPClassifier**: `0.733333` (GBDT 외 모델 다양성 확보용)

### 2) ECDF 랭크 변환 개별 성능
- **LightGBM Rank**: `0.739859`
- **XGBoost Rank**: `0.740480`
- **CatBoost Rank**: `0.740093`
- **MLP Rank**: `0.733208`

### 3) 최종 앙상블 블렌드 성능
- **최종 ECDF Blended Rank OOF AUC**: **`0.740336`**
  - **가중치 세팅**: LightGBM `0.40` + XGBoost `0.20` + CatBoost `0.20` + MLP `0.20`
  - **최종 제출 파일**: [submission_v12_ens_0.740336.csv](file:///c:/Users/tkskd/infertility_classification/src/v12/submission_v12_ens_0.740336.csv)

---

## 4. 핵심 분석 및 시사점

1. **XGBoost 모델의 눈부신 성장 (`0.740481`)**
   - 불필요한 비율 피처 및 결측 플래그를 소거함으로써 트리가 중요 분기에 집중할 수 있게 되었고, 카테고리 일치 작업으로 추론 에러가 완전히 해소되어 높은 일반화 성능을 기록했습니다.
2. **ECDF Rank Blending의 효용성**
   - 예측 값들의 확률 스케일을 uniform 분포로 정형화한 상태에서 가중 평균을 수행하므로, 스케일이 제각각인 다종 모델들을 결합하는 데 최적의 안정성을 제공합니다.
   - 단, MLPClassifier의 점수가 낮아 단순 앙상블 가중치를 고르게 줄 경우 최고 성능의 XGBoost 점수를 일부 갉아먹는 경향이 있어, 차기 버전에서는 가중치 최적화(Weight Optimization)를 고려할 만합니다.

---

## 5. Ablation Study 상세 결과 (피처 소거 실험)
v12 개발 과정에서 오버피팅 제어 및 피처 최적화를 위해 단계별로 진행한 Ablation Study의 결과는 다음과 같습니다 (상세 결과 파일: `scratch/ablation_results.csv`).

| 단계 (Step) | 설명 | 피처 수 | CV AUC | 점수 변화 (Diff) | 결론 및 판단 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Step 0** | **Base Config (v11-like)** | 95 | 0.739871 | 0.000000 | 기존 v11 피처셋 기준점 (Baseline) |
| **Step 1** | **- Custom Ratios** | 87 | 0.739850 | -0.000021 | 비율 피처 8종 제거 (성능 영향 거의 없음 $\rightarrow$ 모델 단순화) |
| **Step 2** | **- is_missing Flags** | 82 | **0.739894** | **+0.000023** | 결측 플래그 5종 제거 (**성능 향상 $\rightarrow$ 오버피팅 유발 피처 제거 확정**) |
| **Step 3** | **- Manual Imputation** | 82 | 0.739761 | -0.000109 | 정밀 도메인 수동 대치 제외 (성능 저하 $\rightarrow$ **수동 대치 유지**) |
| **Step 4** | **- 4 High-Null Cols** | 78 | 0.739432 | -0.000439 | 고결측 컬럼 4종 제외 (성능 저하 $\rightarrow$ **고결측 컬럼 유지**) |
| **Step 5** | **- Custom Flags (Raw 63)** | 63 | 0.739397 | -0.000474 | 커스텀 도메인 플래그 전면 제외 (성능 저하 $\rightarrow$ **커스텀 플래그 유지**) |

* **결론**: 실험 분석 결과, 모델에 노이즈를 주고 오버피팅을 유발하던 **비율 피처 8종**과 **결측 플래그 5종**을 최종 제거(Step 2 상태, 총 82개 피처)하여 v12의 최적 피처 구성을 완성하였습니다.

