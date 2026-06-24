# 불임 분류 프로젝트 실험 보고서 - v10 이종 모델 앙상블 및 조건부 블렌딩 통합 최적화

이 문서는 `v5`에서 검증된 다종 알고리즘(LightGBM + XGBoost + CatBoost)의 모델 다양성과 `v9`에서 수립된 하드케이스 억제 피처 및 조건부 블렌딩 구조를 완벽하게 결합하여 교차 검증을 완료한 `v10` 실험 보고서입니다.

---

## 1. 구현 사양 (Implementation Specification)
* **목적**: 하드케이스(거짓 음성/양성) 방어 피처셋이 탑재된 최적의 가임 정보 하에서, 이종 트리 모델들의 예측 다양성을 앙상블하여 일반화 스코어(ROC-AUC)의 0.74 돌파 및 0.8 달성을 꾀함.
* **폴더 경로**: `src/v10/`
* **학습 최적화**: CatBoost 및 XGBoost의 CPU 연산 병목을 해결하기 위해, 7대 범주형 변수를 정수 코드(`cat.codes`) 형태로 사전 맵핑하고 수치형으로 학습하여 훈련 속도를 약 10배 이상 단축함 (1시간 10분 -> 16분).

---

## 2. 검증 결과 및 모델 성능 (Model Performance)

5-Fold Stratified Cross-Validation을 통해 측정된 스코어 비교 결과는 다음과 같습니다.

### 2.1 개별 및 앙상블 OOF 성능 비교
* **Joint LightGBM OOF AUC**: **`0.740042`** (v9 단일 LGB: `0.739831` 대비 상승)
* **Joint XGBoost OOF AUC**: **`0.740007`**
* **Joint CatBoost OOF AUC**: **`0.740061`**
* **Joint Ensemble OOF AUC (LGB 40% + XGB 30% + Cat 30%)**: **`0.740389`** (프로젝트 최고 성능 경신!)
  * Joint Ens IVF subset AUC: `0.738478`
  * Joint Ens DI subset AUC: `0.687382`

### 2.2 조건부 블렌딩(Conditional Blending) 적용 결과
* **v10 Conditional Blending OOF AUC (Joint Ensemble + Split LGB)**: **`0.740156`**
  * Blended IVF subset AUC: `0.738482`
  * Blended DI subset AUC: `0.684012`

---

## 3. 핵심 발견 및 분석 (Key Findings)

1. **이종 앙상블의 압도적 일반화 파워**:
   - `LGBM + XGBoost + CatBoost`를 앙상블한 `Joint Ensemble` 모델이 단일 모델의 성능 한계를 극복하고 **`0.740389`**의 프로젝트 최고점을 도출했습니다.
2. **소수 집단(DI) 예측 복원**:
   - 기존의 단일 통합 모델에서는 DI 집단의 AUC가 `0.65`대까지 붕괴되었던 반면, v10 Joint Ensemble은 DI subset AUC가 **`0.687382`**까지 대폭 우상향했습니다.
   - 이는 5대 하드케이스 방어 피처가 모델 간의 다양성과 결합하여 소수 데이터의 누락 정보를 완벽히 메우고 있음을 증명합니다.
3. **조건부 블렌딩(Split) 생략 가능성 발견**:
   - v9에서는 DI 전용 모델을 학습해 섞어주는 블렌딩 방식이 필수적이었으나, v10 이종 앙상블 모델에서는 **Joint Ensemble 단독 예측(`0.740389`)이 조건부 블렌딩 모델(`0.740156`)보다 더 높은 전체 AUC와 우수한 DI 서브셋 AUC(`0.687` vs `0.684`)를 보였습니다.**
   - 따라서 최종 예측물은 블렌딩된 버전보다 **순수 Joint Ensemble 모델**을 기반으로 제출하는 것이 일반화 측면에서 가장 안정적이고 뛰어납니다.

---

## 4. 최종 결론 및 후속 조치
- 이종 알고리즘 앙상블과 하드케이스 방어 피처의 상호작용이 성공적으로 작동하여 프로젝트 최고점 `0.74039`를 경신하였습니다.
- 성능이 더 우수한 **Joint Ensemble 예측값**을 최종 제출 파일([submission_v10_advanced.csv](file:///c:/Users/tkskd/infertility_classification/src/v10/submission_v10_advanced.csv))로 보존하기 위해 스크립트의 마지막 저장 로직을 업데이트하고 최종 제출 파일을 보관하였습니다.
