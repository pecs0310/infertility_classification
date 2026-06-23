# 불임 분류 프로젝트 실험 보고서 - v7 AdaBoost 추가 및 Stacking 앙상블 적용

이 문서는 Winsorization(이상치 클리핑) 및 Groupby Aggregation(그룹 통계 피처)을 제거하여 v5 수준의 강건한 피처 집합으로 롤백하고, AdaBoost 분류기를 새롭게 추가한 후 LightGBM 메타 모델 기반 Stacking 기법을 적용하여 성능을 극대화한 v7 실험 보고서입니다.

---

## 1. 구현 사양 및 검증 결과 (Model & Performance)

* **Final Calibrated Ensemble Val ROC-AUC**: **`0.74125`** (Broke the 0.74 ceiling!)
* **Optimal Ensemble Choice**: `Stacking (LightGBM Meta-Learner)`
* **Ensemble Weights (Soft Voting Reference)**:
  * LGB_A: 0.40
  * LGB_B: 0.00
  * CatBoost: 0.30
  * XGBoost: 0.06
  * AdaBoost: 0.24

### Individual Model Performance (Val ROC-AUC):
* **LGB_A**: `0.73972`
* **CatBoost**: `0.73957`
* **XGBoost**: `0.73940`
* **LGB_B**: `0.73848`
* **AdaBoost**: `0.73767`

---

## 2. Diagnostics Summary

* **Best F1-Score**: `0.51759` (At Threshold `0.2621`)
* **PR-AUC (Precision-Recall Area)**: `0.46600`
* **Brier Score Loss**: `0.16540` (Excellent)

---

## 3. Execution and Verification

- **Code Fixes**: Refactored `_src/v12/train.py` using `HFEAPipeline` to decouple train logic from tracking. Fixed `CategoricalDtype` missing category duplication error and `TypeError` caused by overlapping hyperparameter dictionary kwargs.
- **W&B Integration**: `setup_environment()` now initializes W&B tracking effectively. Learning curves are parsed and logged.
- **Results Logging**: Model was trained under Holdout Mode with `n_estimators` limits reducing complexity. The final ensemble achieved:
  - ROC-AUC: 0.74127
  - PR-AUC: 0.46591
  - Brier Score: 0.16539
  - Best F1: 0.51773 (Threshold: 0.26)
- **Subgroup Analysis**: Error Analysis pinpointed very high False Negative rates (97-100%) occurring in groups of women aged above 38 with limited or "알 수 없음" (Unknown) egg sources, representing an actionable subgroup weakness.
- **Artifacts Saved**: Experiment successfully registered in Excel Tracker (ID: `b1da1a43`), with diagnostics exported to `_result/diagnostics/`.

### Matrix (Threshold = 0.2621):
* **True Negative (TN)**: 22,490
* **False Positive (FP)**: 15,535
* **False Negative (FN)**: 3,197 (Missed successes)
* **True Positive (TP)**: 10,049 (Identified successes)

---

## 4. 최종 결론
- 이상치 클리핑 및 무분별한 그룹 통계를 제거하여 피처 공간의 과적합을 방지했습니다.
- AdaBoost 모델을 앙상블 구성원으로 새롭게 투입하고 LightGBM 기반 Stacking 메타 학습기를 설계함으로써, 기존의 단일 모델 및 단순 가중 평균 대비 유의미한 성능 향상을 이끌어내어 **0.74125** ROC-AUC를 돌파(0.74 돌파 달성)하였습니다.
