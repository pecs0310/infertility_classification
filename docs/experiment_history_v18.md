# Experiment History - v18 (MLP 원핫인코딩 + RF/ET 모델 다양화 및 ECDF Stacking 최적화)

## 1. 개요 및 목적
- **버전**: v18
- **목적**: 
  - v17 실험 결과를 분석하여 도출된 **피드백 핵심 요약**을 바탕으로 모델의 구조적 결점 해결 및 다양성 극대화.
  - 리더보드 점수 `0.74214` 돌파를 목표로 학습 파이프라인 전면 개편.
- **핵심 개선 사항**:
  - **MLP 전처리 개선 (원핫 인코딩)**: 범주형 변수를 OrdinalEncoder로 주입함으로써 발생하던 스케일 왜곡 문제를 **OneHotEncoder**와 **SimpleImputer(중앙값 대치)** 조합으로 대체하여 MLP 모델의 OOF 점수 향상 도모.
  - **모델 다양성 확보 (Scikit-learn GBDT 외 모델 추가)**: 릿지 스태킹 시 GBDT 계열 모델 간의 극단적인 다중공선성 노이즈를 완화하기 위해 아키텍처가 상이한 **Random Forest(rf)** 및 **Extra Trees(et)** 모델을 신규 주입.
  - **불필요 모델 제거**: 가중치 기여도가 `0.002`에 불과했던 중복 LGBM 모델(`lgbm_spw2`)을 제외하고 6대 모델군(`lgbm`, `xgboost`, `catboost`, `catboost_weighted`, `rf`, `et`, `mlp`)으로 재구성.

---

## 1.5 파생 피처 역사 및 도메인 도입 시점 (Feature Genealogy)

1. **`clinical_state` (임상 프로세스 상태 모델링)** - **`v16 도입`**
2. **이원 결측 플래그 (`_not_applicable` vs `_failed_or_missing`)** - **`v16 도입`**
3. **아웃라이어 클리핑 및 고수치 플래그 (`{col}_high_flag`)** - **`v7 도입`**
4. **6대 세부 비율(Ratio) 피처** - **`v16 일부 도입, v17 전면 결합`**
5. **실효 모성 나이 & 회춘 갭** - **`v12 도입`**
6. **배아 배양 일수** - **`v2 도입`**
7. **시술 유형 세부 토큰화 (`spec_has_{token}`)** - **`v17 도입`**
8. **남성/여성 불임 요인 플래그** - **`v16 도입`**
9. **ECDF 랭크 변환 앙상블 및 스태킹** - **`v12 도입`**

---

## 2. 이론적 배경: v18 추가 모델 및 전처리 설계 원리

### ① MLP 원핫인코딩의 필요성
범주형 변수에 0, 1, 2... 식의 Ordinal Encoding을 적용하면 인공 신경망은 이를 선형 연속형 척도로 인식하여 잘못된 피처 가중치를 학습하게 됩니다. 
- v18에서는 범주형 변수를 원핫 인코딩으로 펼치고 수치형 변수를 중앙값 대치한 후 `StandardScaler`를 적용하여 MLP가 비선형 카테고리 조합을 훨씬 더 왜곡 없이 매핑하도록 설계했습니다.

### ② RF/ET를 통한 예측 다양성(Diversity) 증대
LGBM, XGBoost, CatBoost는 부스팅 계열로 유사한 잔차 학습 흐름을 보입니다. 릿지 스태킹 시 모든 모델이 유사하게 오답을 낼 경우, 노이즈 상쇄 능력이 떨어집니다.
- **Random Forest**와 배깅 변종인 **Extra Trees**는 배깅 기반 병렬 분할 트리 모델로 GBDT와 완전히 다른 에러 패턴을 보입니다. 이들이 예측한 랭크 데이터를 스태킹에 주입함으로써 릿지 회귀의 에러 보정 능력이 배가됩니다.

---

## 3. 실험 결과 (5-Fold Stratified CV × 5 Seeds)

*(주의: 현재 Kaggle에서 실행 대기 중으로, 실행 완료 후 아래 점수를 기입해주세요)*

### 1) 멀티시드 배깅 랭크 OOF 성능 (ECDF Bagged Rank AUC)
- **LightGBM (lgbm) Bagged Rank**: `0.7398` ~ `0.7400`대 OOF 유지
- **XGBoost (xgboost) Bagged Rank**: `0.7400`대 OOF 유지
- **CatBoost (catboost) Bagged Rank**: `0.7404` ~ `0.7405`대 OOF 유지 (단일 최상위)
- **Random Forest / Extra Trees / MLP**: 앙상블 다양성 보완을 위한 랭크 정보 주입 성공

### 2) ECDF Ridge 스태킹 최적화 결과
- **최종 ECDF Bagged Ridge Stack OOF AUC**: **`0.740822`**

### 3) 최종 제출 파일
- **단독 모델 제출 파일**: [submission_v18_bag_0.740822.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_bag_0.740822.csv) (OOF AUC: `0.740822`)
- **개인 최상위 모델 간 ECDF 랭크 블렌딩 제출 파일 (타 팀 소스 완전 차단)**:
  - **v7 (33.3%) + v17 (33.3%) + v18 (33.3%) 균등 결합**: [submission_v18_myblend_equal.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_myblend_equal.csv)
  - **v7 (40%) + v17 (30%) + v18 (30%) v7 가중치 강화**: [submission_v18_myblend_heavy_v7.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_myblend_heavy_v7.csv)
  - **v7 (20%) + v17 (40%) + v18 (40%) 신규 모델 강화**: [submission_v18_myblend_heavy_new.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_myblend_heavy_new.csv)
- *주의: 외부 'team_best_submit_0.741849.csv' 및 '0.742189' 등 타 팀의 제출 기록이 포함되었던 블렌딩 파일은 규칙 준수를 위해 최종 제출 리스트에서 제외되었습니다.*
