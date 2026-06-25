# Experiment History - v17 (임상 상태 모델링 + Rich Feature + ECDF Ridge 스태킹 통합 최적화)

## 1. 개요 및 목적
- **버전**: v17
- **목적**: 
  - 사용자님의 정밀한 **v16 임상 상태 모델링(clinical_state)** 및 **이원 결측 플래그** 기능과 동료분의 **Rich Features (Keep-NaN, 클리핑, 6대 비율, 시술 토큰화, train 카테고리 기준 고정)**를 하나의 단일 피처셋으로 완벽 통합.
  - 단순 가중평균 앙상블의 한계를 넘기 위해, **ECDF rank 변환 후 Ridge/Linear Regression 메타러너 스태킹**을 도입하여 모델 간의 다중공선성 노이즈를 수학적으로 상쇄하고 점수 극대화 달성.
- **핵심 설계 및 적용 사항**:
  - **피처셋 병합**: v16의 7단계 `clinical_state`와 결측 성격 분리 플래그 + 동료분의 Keep-NaN(구조적 결측 보존), 9개 변수 Clip 규칙 및 High Flag 생성, 6대 임상 안전 비율 피처, 10종 시술 상세 토큰화 반영.
  - **DACON 누수 규정 완벽 준수**: 카테고리 고유값 풀을 Train 데이터 기준으로만 고정하고, Test에 새로 나타나거나 결측인 범주는 `정보없음` 센티널 문자열로 안전하게 흡수.
  - **후보 모델군 확장**: 새로운 피처셋을 기반으로 `LGBM`, `XGBoost`, `CatBoost (Native Category)`, `CatBoost (Weighted)`, `LGBM_SPW2`, `MLP` 총 6대 후보 모델 정의.
  - **5-시드 5-Fold Stratified CV 배깅 및 스태킹**: 150회 교차 검증 학습을 거쳐 수집한 OOF 예측으로 ECDF 스태킹 메타러너(Ridge/Linear Regression) 최적화 수행.

---

## 1.5 파생 피처 역사 및 도메인 도입 시점 (Feature Genealogy)

본 실험에 사용된 병합 피처셋의 각 요소들이 처음 설계되고 도입된 버전 히스토리는 다음과 같습니다:

1. **`clinical_state` (임상 프로세스 상태 모델링)** - **`v16 도입`**
   - 시술 유형과 진행 단계별로 환자의 현재 주기를 7가지 계층적 상태로 세분화.
2. **이원 결측 플래그 (`_not_applicable` vs `_failed_or_missing`)** - **`v16 도입`**
   - 단순 결측치 처리를 넘어, 시술 구조상 발생한 MAR 결측과 기록 누락/실패로 인한 MNAR 결측을 논리적으로 분리.
3. **아웃라이어 클리핑 및 고수치 플래그 (`{col}_high_flag`)** - **`v7 도입`**
   - 생성 배아 수, 수집 난자 수 등 카운트 피처의 과도한 아웃라이어 극단치로 인한 학습 노이즈를 제어하기 위한 클리핑 기법.
4. **6대 세부 비율(Ratio) 피처** - **`v16 일부 도입, v17 전면 결합`**
   - `fertilization_rate` 등 단순 효율 피처 2종은 **v16**에 기입되었으나, 수정율/이식율/동결율/해동배아이식율 등 세분화된 6종의 안전 비율(`safe_ratio`) 피처셋은 **v17**에서 최종적으로 동료분 피처셋과 병합되었습니다.
5. **실효 모성 나이 (`effective_maternal_age_mid`) & 회춘 갭 (`donor_rejuvenation_gap`)** - **`v12 도입`**
   - 기증 난자 사용 시 생물학적 나이 보정과 나이 극복 수치를 반영하기 위해 **v12**에서 최초 도입 및 결합.
6. **배아 배양 일수 (`embryo_culture_days`)** - **`v2 도입`**
   - 배아의 이식 경과일과 혼합 경과일 차이를 구해 배아의 이식 단계(3일 분할기 vs 5일 포배기)를 모델이 파악하도록 지원.
7. **시술 유형 세부 토큰화 (`spec_has_{token}`)** - **`v17 도입`**
   - 텍스트 형태의 시술명에서 포배기 배아 이식(BLASTOCYST) 및 보조 부화술(AH) 등 핵심 기술 키워드를 추출하여 바이너리화.
8. **남성/여성 불임 요인 플래그 (`is_male_infertility`, `is_female_infertility`)** - **`v16 도입`**
   - 수십 개로 희소하게 쪼개져 있던 불임 원인 변수들을 남성 측 요인과 여성 측 요인으로 그룹화하여 정보 집중도 향상.
9. **ECDF 랭크 변환 앙상블 및 스태킹 (ECDF Rank Blending & Stacking)** - **`v12 도입`**
   - 모델 간의 예측 스케일 불일치 문제를 해결하기 위해 예측값을 백분위수로 변환하는 기법. **v12**에서 고정 가중치 블렌딩으로 처음 적용된 후, **v13** Nelder-Mead 최적화, **v14** 5-시드 배깅, 그리고 **v17** Ridge 스태킹으로 발전하며 본 프로젝트의 앙상블 뼈대를 이루었습니다.

---

## 2. 이론적 배경: ECDF Ridge 스태킹의 성능 향상 및 과적합 방지 원리

### ① 상관 노이즈 상쇄 (Noise Cancellation)
LightGBM, XGBoost, CatBoost 등 동일한 학습 데이터를 사용한 기반 모델들은 예측값 간에 매우 강한 상관관계를 가집니다. 즉, **오답을 낼 때 같이 틀리는 다중공선성(Multicollinearity) 노이즈**를 공유합니다.
- **단순 가중평균(v16)**은 가중치가 모두 양수($w_i \ge 0$)이고 합이 1이어야 하므로, 단순히 예측 분산을 줄일 뿐 상관 노이즈를 차감하지 못합니다.
- **선형회귀 스태킹(v17)**은 계수($\beta_i$)에 **음수(Negative) 가중치**를 부여할 수 있습니다. 예를 들어 $1.2 \times \text{Model A} - 0.3 \times \text{Model B}$와 같은 결합을 통해 두 모델이 공유하는 과적합 노이즈를 수학적으로 **'빼주기 연산(Subtraction)'**하여 완벽하게 상쇄합니다.

### ② ECDF 변환을 통한 랭크 스케일 통일 (Normalization)
모델마다 출력하는 확률값의 원본 스케일과 분포는 완전히 다릅니다. (예: 가중치가 균형 잡힌 CatBoost는 확률값 평균이 높고, LGBM은 0과 1 근처에 쏠림)
- 이를 그대로 선형 모델에 넣으면 특정 모델의 극단적 분포에 왜곡되어 오버피팅이 발생합니다.
- **ECDF(경험적 누적분포함수)** 변환은 모든 모델의 예측 확률값을 **0~1 사이의 백분위수(Percentile, Rank)**로 변환하여 동일한 체급(Scale)으로 통일합니다. 이 상태에서 표준화(`StandardScaler`) 후 회귀에 적용하므로 최적화가 극도로 안정적입니다.

### ③ OOF (Out-of-Fold) 예측을 통한 과적합 원천 차단
학습 데이터 자체에 과적합(Overfitting)될 위험은 데이터셋이 Train과 Test로 나뉘는 모든 대회에서 존재합니다. Train과 Test가 아무리 비슷한 분포에서 나왔더라도, **무작위 샘플링 노이즈는 서로 다르기 때문**에 Train에 과적합된 모델은 실전에서 무조건 탈락합니다.
- 스태킹을 진행할 때 기반 모델들의 학습 데이터 내부 예측값(In-sample)으로 학습하면, 메타 모델은 "기반 모델들이 100% 완벽하다"고 착각하여 오차 보정법을 배우지 못합니다.
- 스태킹에 반드시 **OOF(Out-of-Fold) 예측값**을 사용함으로써, 메타 모델은 **"기반 모델들이 처음 보는 신규 데이터를 예측할 때 저지르는 실수 패턴"**을 학습하게 됩니다. 따라서 테스트 데이터를 맞닥뜨렸을 때도 흔들림 없이 높은 점수를 유지할 수 있습니다.
- 또한 **Ridge 메타러너(L2 정규화)**를 적용하여 모델 간 다중공선성으로 인해 계수가 $+10.0$, $-9.5$와 같이 이상 비대해지는 현상을 방지하고 일반화 성능을 한 번 더 잠급니다.

---

## 3. 실험 결과 (5-Fold Stratified CV × 5 Seeds)

본 실험은 5개 시드(42, 1004, 7, 2026, 88)와 5-Fold Stratified CV를 통해 학습을 진행하였으며, 각 시드별 checkpoint를 저장 후 병합 및 ECDF Stacking을 진행했습니다.

### 1) 멀티시드 배깅 랭크 OOF 성능 (ECDF Bagged Rank AUC)
- **LightGBM (lgbm) Bagged Rank**: `0.740332`
- **XGBoost (xgboost) Bagged Rank**: `0.740535`
- **CatBoost (catboost) Bagged Rank**: `0.740742`
- **CatBoost Weighted Bagged Rank**: `0.740737`
- **LightGBM SPW2 Bagged Rank**: `0.737209`
- **MLP Classifier Bagged Rank**: `0.737488`

### 2) ECDF Ridge 스태킹 최적화 결과
- **Ridge Alpha 그리드 탐색**:
  - `alpha = 0.0 (LR)`: `0.740828`
  - `alpha = 0.1`: `0.740828`
  - `alpha = 0.3`: `0.740828`
  - `alpha = 0.5`: `0.740828`
  - `alpha = 1.0`: `0.740828`
  - `alpha = 3.0`: `0.740828`
  - `alpha = 10.0`: `0.740828`
  - `alpha = 30.0`: `0.740829`
  - `alpha = 100.0`: `0.740830`
- **최적 Alpha**: `100.0`
- **최종 ECDF Bagged Ridge Stack OOF AUC**: **`0.740830`**

### 3) 최종 제출 파일
- **최종 제출 파일**: [submission_v17_bag_0.740830.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v17_bag_0.740830.csv) (OOF AUC: `0.740830`)
