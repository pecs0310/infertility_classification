# Experiment History - v14 (ECDF 최적 가중치 랭크 블렌딩 + 멀티시드 배깅)

## 1. 개요 및 목적
- **버전**: v14
- **목적**: `v13` 모델 가중치 최적화(ECDF Rank Blending) 파이프라인에 **5-시드 멀티배깅(Multi-seed Bagging)**을 통합 적용하여 단일 시드의 노이즈를 억제하고 전체 모델의 일반화 스코어(ROC-AUC) 극대화.
- **주요 개선 사항**:
  - GBDT 3종(LightGBM, XGBoost, CatBoost)과 MLPClassifier에 대해 5개 시드(`[42, 1004, 7, 2026, 88]`)로 5-Fold Stratified CV(총 25회 학습)를 수행하여 예측 분포를 다양화.
  - CatBoost의 CPU 훈련 병목 해결을 위해 **정수 코드 인코딩(Ordinal Encoding)**으로 변환하여 학습하여 훈련 속도를 10배 이상 단축시킴 (학습 시간 단축 성공).
  - 시드/폴드 내부에서 개별적으로 ECDF 랭크 변환한 후 시드 간 평균(Seed Averaging)을 적용함으로써 데이터 누수 없는 완벽한 배깅 구현.

---

## 2. 모델 및 학습 설정 (5-Fold Stratified CV × 5 Seeds)
- **피처 셋**: v12/v13에서 최적화된 81개 피처셋 유지 (실질 가임 연령 및 리쥬브네이션 갭 피처 포함).
- **시드 목록**: `[42, 1004, 7, 2026, 88]`
- **개별 모델 구성**:
  1. **LightGBM**: Tuned Joint LGBM (시드별 OOF AUC 평균: `0.739815`)
  2. **XGBoost**: Tuned Joint XGBoost (시드별 OOF AUC 평균: `0.740306`)
  3. **CatBoost**: Joint CatBoost (Ordinal Encoded, 시드별 OOF AUC 평균: `0.739900`)
  4. **MLPClassifier**: Neural Network (시드별 OOF AUC 평균: `0.732784`)

---

## 3. 실험 결과 (ROC-AUC)

### 1) 멀티시드 배깅 랭크 OOF 성능 (ECDF Bagged Rank AUC)
- **LightGBM Bagged Rank**: `0.740247` (단일 시드 대비 성능 대폭 상승)
- **XGBoost Bagged Rank**: `0.740592`
- **CatBoost Bagged Rank**: `0.740184`
- **MLP Classifier Bagged Rank**: `0.735767`

### 2) 가중치 최적화 결과 (Nelder-Mead on Bagged Ranks)
최적화 과정을 통해 산출된 배깅 랭크용 최적 가중치는 다음과 같습니다.
- **LightGBM Weight**: `0.1774`
- **XGBoost Weight**: 🏆 `0.6521` (가장 우수한 성능의 XGBoost에 높은 비중 집중)
- **CatBoost Weight**: `0.1064`
- **MLPClassifier Weight**: `0.0641` (일반화 성능 보완용 최소 가중치 배분)

### 3) 최종 앙상블 블렌드 성능
- **최종 Optimized ECDF Bagged Rank OOF AUC**: **`0.740645`**
  - **`v13` 단일 시드 최적화 (`0.740534`) 대비 `+0.000111` 향상**으로 단일 파이프라인 최고점 경신.
- **최종 제출 파일**: [submission_v14_bag_0.740645.csv](file:///C:/Users/tkskd/infertility_classification/src/v14/submission_v14_bag_0.740645.csv)
- **백업 제출 파일**: [submission_v14_bag_0.740645.csv](file:///C:/Users/tkskd/infertility_classification/data/submission_v14_bag_0.740645.csv)

---

## 4. 핵심 분석 및 시사점
1. **멀티시드 배깅의 일반화 성능 실증**:
   - 단일 시드 대비 LightGBM의 랭크 OOF가 `0.7398` $\rightarrow$ `0.7402`로 대폭 향상되었고, XGBoost 역시 `0.74059`로 상승하여 멀티시드 배깅의 강점을 다시금 확인했습니다.
2. **최적 가중치 쏠림 현상 극대화**:
   - 배깅을 거친 후 XGBoost가 압도적으로 뛰어난 일반화 패턴을 보여, Nelder-Mead 최적화가 XGBoost에 무려 `65.2%`에 달하는 높은 가중치를 배분했습니다.
3. **CatBoost 인코딩 최적화 성공**:
   - CatBoost에 정수 코드 인코딩을 적용해 학습 속도를 10배 이상 단축시킴으로써, 총 5개 시드(25회 학습)에 달하는 방대한 파이프라인 연산을 안정적으로 빠르게 완료했습니다.
