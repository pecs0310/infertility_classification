# Experiment History - v13 (ECDF Rank Blending 가중치 최적화)

## 1. 개요 및 목적
- **버전**: v13
- **목적**: `v12` 단일 GBDT 3종(LightGBM, XGBoost, CatBoost)과 MLPClassifier의 앙상블 가중치를 Scipy의 Nelder-Mead 최적화 도구를 사용하여 자동 튜닝.
- **주요 개선 사항**: 
  - 기존의 임의 고정 가중치 `[0.40, 0.20, 0.20, 0.20]` 방식 대신, ECDF 랭크 변환된 OOF 예측값을 바탕으로 OOF ROC-AUC를 최대화하는 가중치를 동적으로 탐색.
  - 가중치가 음수가 되지 않고 합이 1.0이 되도록 Softmax 매핑 함수를 목적 함수에 적용하고, Gradient-free 방식인 Nelder-Mead 알고리즘을 활용하여 랭크 메트릭의 최적 가중치를 안정적으로 수렴시킴.

---

## 2. 모델 및 학습 설정 (5-Fold Stratified CV)
- **피처 셋**: `v12`에서 확정된 최적 피처 81개 유지 (실질 가임 연령 및 리쥬브네이션 갭 피처 포함, 노이즈 유발 비율 피처/결측 플래그 제거).
- **개별 모델 구성**:
  1. **LightGBM**: Tuned Joint LGBM (AUC: `0.739860`)
  2. **XGBoost**: Tuned Joint XGBoost (AUC: `0.740481`)
  3. **CatBoost**: Joint CatBoost (AUC: `0.740098`)
  4. **MLPClassifier**: Neural Network (AUC: `0.733333`)

---

## 3. 실험 결과 (ROC-AUC)

### 1) 개별 모델 랭크 변환 OOF 성능 (ECDF Rank AUC)
- **LightGBM Rank**: `0.739859`
- **XGBoost Rank**: `0.740480`
- **CatBoost Rank**: `0.740093`
- **MLP Rank**: `0.733208`

### 2) 가중치 최적화 결과 (Nelder-Mead + Softmax)
최적화 과정을 통해 산출된 모델별 가중치는 다음과 같습니다.
- **LightGBM Weight**: `0.3432` (기존 `0.40`)
- **XGBoost Weight**: 🏆 `0.3496` (기존 `0.20` $\rightarrow$ 최고 단일 모델 가중치 대폭 반영)
- **CatBoost Weight**: `0.2263` (기존 `0.20`)
- **MLPClassifier Weight**: 📉 `0.0809` (기존 `0.20` $\rightarrow$ 낮은 성능의 MLP 비중을 낮춤)

### 3) 최종 앙상블 블렌드 성능
- **최종 Optimized ECDF Blended Rank OOF AUC**: **`0.740534`**
  - **`v12` 고정 가중치 블렌드 (`0.740336`) 대비 `+0.000198` 향상** 및 **최고 단일 모델 XGBoost (`0.740481`) 대비 `+0.000053` 향상**으로 전체 최고 스코어 경신.
- **최종 제출 파일**: [submission_v13_ens_0.740534.csv](file:///C:/Users/tkskd/infertility_classification/src/v13/submission_v13_ens_0.740534.csv)
- **백업 제출 파일**: [submission_v13_ens_0.740534.csv](file:///C:/Users/tkskd/infertility_classification/data/submission_v13_ens_0.740534.csv)

---

## 4. 핵심 분석 및 시사점
1. **가중치 최적화의 성공성**:
   - 예측력이 상대적으로 떨어지는 MLPClassifier의 가중치가 자동으로 `0.0809`까지 감소한 반면, 최고 성능의 XGBoost 가중치가 `0.3496`으로 비중이 늘어나면서 단순 평균 방식의 점수 하락 문제를 극복했습니다.
2. **ECDF 블렌딩의 조화**:
   - GBDT 3종과 MLPClassifier가 랭크 단위로 조화롭게 결합하여 단일 최고 성능인 XGBoost의 성능을 뛰어넘는 `0.740534`를 도출하여 일반화 안정성을 입증하였습니다.
3. **학습 속도 캐싱**:
   - OOF 예측값 캐시(`src/v13/checkpoints/model_oofs.pkl`)가 성공적으로 생성되어, 차후 가중치 탐색 수식이나 최적화 옵션을 다르게 테스트할 때 모델 학습 단계를 스킵하고 수 초 만에 빠르게 가중치 튜닝을 재실행할 수 있는 인프라가 마련되었습니다.
