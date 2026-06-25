# Experiment History - v19 (MLP 규제 강화 + 탐욕적 모델 선택 ECDF Ridge Stacking + Optuna 최적화)

## 1. 개요 및 목적
- **버전**: v19
- **목적**: 
  - v18 실험의 미세한 점수 하락(0.740830 -> 0.740822)을 극복하고 리더보드 최고점을 향해 최적화.
  - 범주형 원핫인코딩으로 고차원화된 MLP 모델의 과적합 제어.
  - 스태킹 메타러너 학습 시, 성능이 다소 떨어져 노이즈를 유발할 수 있는 모델을 자동으로 차단하는 탐욕적 모델 선택 도입.
  - 최상위 단독 모델인 CatBoost의 하이퍼파라미터 튜닝을 위한 Optuna 튜닝 도구 구축.

## 2. 핵심 개선 사항
1. **MLP L2 규제 상향 (Overfitting Control)**:
   - 범주형 원핫인코딩에 의해 입력 차원이 크게 팽창하면서 가중치 파라미터가 급증하여 발생하던 과적합을 방지하기 위해 L2 Penalty 계수(`alpha`)를 기존 `1e-4`에서 `0.01`로 대폭 상향 조정.
2. **탐욕적 스태킹 모델 선택 (Greedy Stacking Model Selection)**:
   - ECDF Ridge Stacking 학습 시 7개 모델을 맹목적으로 모두 사용하는 대신, OOF AUC 성능을 향상시키는 데 기여하는 최적의 모델 서브셋만을 단계적으로 선택하여 결합하도록 알고리즘 구현.
3. **CatBoost Optuna 자동 하이퍼파라미터 튜닝**:
   - GBDT 계열 중 가장 강력한 단독 최상위 모델인 CatBoost의 `learning_rate`, `depth`, `l2_leaf_reg`를 GPU 상에서 최적화하도록 Optuna 프레임워크 연동.

## 3. 학습 및 검증 아키텍처 (5-Fold CV × 5 Seeds)
- **학습 코드**: [train_v19_opt_gpu.py](file:///c:/Users/tkskd/infertility_classification/src/v19/train_v19_opt_gpu.py)
- **Optuna 코드**: [optuna_tune_gpu.py](file:///c:/Users/tkskd/infertility_classification/src/v19/optuna_tune_gpu.py)

---

## 4. 예상 효과 및 행동 가이드
- 탐욕적 스태킹 선택을 통해 성능 기여도가 없거나 노이즈가 되는 `Extra Trees` 및 `Random Forest` 일부 변동값들이 제거되어 스태킹 OOF 점수가 v17 최고점인 `0.740830`을 안정적으로 경신할 것으로 기대됩니다.
- Kaggle 환경에서 `train_v19_opt_gpu.py`를 실행하여 생성되는 `submission_v19_opt_*.csv` 파일을 최종 리더보드에 제출하는 방식으로 이 하락을 극복합니다.
