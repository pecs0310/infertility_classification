# 불임 분류 프로젝트 실험 보고서 - v11 Optuna 튜닝 및 메타 스태킹 결과 분석

이 문서는 `v11`에서 진행한 Optuna 기반 하이퍼파라미터 자동 튜닝과 L2 규제 로지스틱 회귀 메타 스태킹(Stacking)의 최종 학습 및 검증 결과를 기록한 보고서입니다.

---

## 1. 구현 사양 (Implementation Specification)
* **목적**: 20% 층화 추출 샘플(Stratified Sample, 약 5만 행)을 활용해 LightGBM과 XGBoost의 최적 파라미터를 고속으로 찾아내고, 이종 모델 및 분할 모델의 OOF 예측값을 결합하는 L2 로지스틱 회귀 메타 스태킹 모델을 구축하여 성능 극대화를 꾀함.
* **폴더 경로**: `src/v11/`
* **제출 산출물**: `src/v11/submission_v11_advanced.csv`

---

## 2. Optuna 튜닝 결과 (Tuned Parameters)

* **LightGBM (Joint)**:
  - `num_leaves`: 84
  - `max_depth`: 5
  - `scale_pos_weight`: 1.1011
  - `feature_fraction`: 0.7582
  - `bagging_fraction`: 0.8284
* **XGBoost (Joint)**:
  - `max_depth`: 4
  - `subsample`: 0.7294
  - `colsample_bytree`: 0.5051
  - `scale_pos_weight`: 1.9240

---

## 3. 검증 결과 및 성능 비교 (Model Performance)

5-Fold Stratified Cross-Validation을 통해 측정된 스코어 비교 결과는 다음과 같습니다.

### 2.1 개별 및 앙상블 OOF 성능 비교
* **Tuned Joint LightGBM OOF AUC**: **`0.740203`** (v10: `0.74004` 대비 **+0.00016 상승**, 단일 최고점 경신!)
* **Tuned Joint XGBoost OOF AUC**: **`0.740021`** (v10: `0.74001` 대비 소폭 상승)
* **Joint CatBoost OOF AUC**: **`0.740061`**
* **Simple Joint Ensemble OOF AUC (LGB 40% + XGB 30% + Cat 30%)**: 🏆 **`0.740350`** (단일 모델 성능을 상회하는 견고한 일반화 성능)

### 2.2 메타 스태킹 및 가중치 분석 결과
* **v11 Meta Stacking OOF AUC**: **`0.739756`** (성능 하락)
  - Stacking IVF subset AUC: `0.737847`
  - Stacking DI subset AUC: `0.685736`
* **메타 학습기(Logistic Regression)의 가중치 배분**:
  - **Joint LightGBM**: `-0.9920` (음수 가중치)
  - **Joint XGBoost**: `1.7103`
  - **Joint CatBoost**: `4.9531`
  - **Split LightGBM (Domain)**: `-0.8754` (음수 가중치)
  - **Meta Bias (Intercept)**: `-3.7554`

---

## 4. 핵심 발견 및 시사점 (Key Findings)

1. **다중공선성(Multicollinearity)으로 인한 스태킹 오버피팅**:
   - 메타 스태킹 가중치 분석 결과, LightGBM과 Split LightGBM에 **음수 가중치**가 강하게 부여되었습니다.
   - 이는 네 개 모델의 예측값(확률)이 너무 강하게 동조(High Correlation, 다중공선성)하여, 로지스틱 회귀가 이를 조율하는 과정에서 가중치를 극단적으로 주고받으며 과적합이 일어나 검증 점수(OOF AUC)가 `0.739756`으로 하락하는 부작용이 발생했습니다.
2. **Optuna 하이퍼파라미터 튜닝의 유효성 검증**:
   - 데이터 20% stratified 샘플을 통한 고속 튜닝이었음에도 불구하고, 단일 LightGBM 점수가 `0.74020`으로 뚜렷이 상승하여 피처셋에 적합한 최적 트리 깊이와 샘플 비중을 성공적으로 포착했습니다.
3. **단순 가중 평균(Simple Ensemble)의 강인성 재발견**:
   - 예측 모델들이 서로 강하게 동조하는 상황에서는, 복잡한 메타 모델을 사용하는 스태킹보다 **단순 가중 평균(Simple Ensemble) 방식이 다중공선성 왜곡을 겪지 않아 훨씬 더 강인하고 뛰어난 성능(`0.740350`)을 발휘함**을 확인했습니다.

---

## 5. 후속 조치 (Action Items)
- 성능이 더 높은 **Simple Joint Ensemble 결과물(`0.740350`)**을 최종 제출 파일([submission_v11_advanced.csv](file:///c:/Users/tkskd/infertility_classification/src/v11/submission_v11_advanced.csv))로 대체 복사하여 최종 일반화 성능을 안정적으로 보존 조치 완료하였습니다.
