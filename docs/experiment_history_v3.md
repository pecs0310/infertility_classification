# 불임 분류 프로젝트 실험 보고서 - v3 Tuning & Ensemble

이 문서는 도메인 피처 엔지니어링이 완료된 데이터셋 위에 하이퍼파라미터 최적화와 3대 GBDT 알고리즘(LightGBM, XGBoost, CatBoost) 가중 평균 앙상블을 적용하여 프로젝트의 최상위 성능(v3)을 기록한 보고서입니다.

---

## 1. 구현 사양 (Implementation Specification)
* **목적**: 개별 트리 모델의 구조적 한계와 과적합을 제어하고, 이종 모델의 다양한 훈련 분기를 소프트 보팅하여 일반화 성능을 극대화함.
* **폴더 경로**: `src/v3/`
* **제출 산출물**: `src/v3/submission_v3_ensemble.csv`
* **추가 필수 패키지**: `xgboost`, `catboost`

---

## 2. 전처리 고도화 (Preprocessing)
* **범주형 피처 완벽 동기화 (Sync Categories)**:
  - XGBoost가 Pandas `category` 데이터 타입을 처리할 때, 훈련 데이터와 평가 데이터 간 범주가 일치하지 않는 예외 상황(`Found a category not in the training set`)을 원천 차단하기 위해 전처리 함수에 train/test의 category 고유값 집합을 병합하여 일치시키는 로직 구현.
  - 범주형 결측을 사전에 `'Missing'` 문자열로 완벽히 방어 처리하여 CatBoost와 XGBoost의 훈련 중단 문제 해결.

---

## 3. 모델 개선 및 앙상블 아키텍처 (Model Improvement & Ensemble)

### 3.1 LightGBM 하이퍼파라미터 최적화 (Tuned LightGBM)
- 트리 과적합 제어 및 세밀한 탐색을 위해 튜닝 진행:
  - `learning_rate`: 0.05 $\rightarrow$ **0.02** (하향)
  - `num_leaves`: 31 $\rightarrow$ **63** (용량 증가)
  - `max_depth`: 제한 없음 $\rightarrow$ **9** (트리 깊이 제한)
  - `scale_pos_weight`: 1.0 $\rightarrow$ **1.5** (클래스 불균형에 가중치 부여)
  - `feature_fraction`: 0.8 $\rightarrow$ **0.7** (피처 샘플링을 통한 정규화)

### 3.2 이종 분류 모델 결합
1. **Model 1: LightGBM (OOF AUC: 0.73994)**
   - 파라미터 최적화로 정규화 및 수용량 향상.
2. **Model 2: XGBoost (OOF AUC: 0.74013)**
   - `enable_categorical=True` 및 `tree_method='hist'`를 적용해 Pandas 카테고리 데이터 최적 학습. 정교한 L1/L2 규제 적용.
3. **Model 3: CatBoost (OOF AUC: 0.73974)**
   - 범주형 특성 전처리가 매우 강력한 CatBoost 모델을 결합해 다른 형태의 트리 구조 다양성 확보.

### 3.3 가중 평균 앙상블 (Weighted Average Soft Voting)
- 세 모델의 OOF 성능 편차가 크지 않고 우수함을 감안하여 최종 가중 합산:
  $$\text{Final Pred} = 0.40 \times \text{LightGBM} + 0.30 \times \text{XGBoost} + 0.30 \times \text{CatBoost}$$

---

## 4. 검증 결과 (Performance)
* **교차 검증 ROC-AUC 스코어 (앙상블)**:
  - **LightGBM OOF**: `0.73994`
  - **XGBoost OOF**: `0.74013`
  - **CatBoost OOF**: `0.73974`
  - **최종 앙상블 OOF 스코어**: **`0.74031`** (이전 v2 대비 **`+0.00017` 성능 향상 및 프로젝트 최고점**)

---

## 5. 결론 및 향후 과제
- 파라미터 미세 튜닝과 이종 모델의 앙상블이 결합되면서 단일 모델이 갖는 변동 폭을 상쇄하여 최종 OOF 점수가 눈에 띄게 개선되었습니다.
- 본 버전의 출력물 `submission_v3_ensemble.csv`이 불임 성공 여부 분류의 최종 제출 사양으로 권장됩니다.
