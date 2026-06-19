import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 가상환경 내부로 진입하면 실행 스크립트(run_v1.ps1)에 의해 이미 lightgbm이 설치되어 있을 것이지만,
# 혹시 개별적으로 실행할 경우를 대비하여 패키지를 임포트합니다.
import lightgbm as lgb

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print(f"Loading data from: {DATA_DIR}")

# 2. 데이터 로드
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUB_PATH)

target_col = '임신 성공 여부'
features = [col for col in train.columns if col not in ['ID', target_col]]

X = train[features].copy()
y = train[target_col]
X_test = test[features].copy()

# 3. 전처리: LightGBM에 적합하도록 범주형 변수를 'category' 타입으로 변환
cat_cols = []
for col in features:
    # numeric 혹은 bool type이 아니면 category로 변환 (pandas 3.x의 string 타입 대응)
    if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        cat_cols.append(col)

print(f"Total features: {len(features)}")
print(f"Categorical features (converted to pandas category): {len(cat_cols)}")

# 4. Stratified K-Fold Validation (ROC-AUC)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\n--- Starting K-Fold Validation (LightGBM v1 Train) ---")
fold_scores = []

# LightGBM의 warning이나 log 출력을 제어
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Dataset 객체 생성 (LightGBM native 처리용)
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    # LightGBM 학습 (Early Stopping 적용)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks
    )
    
    # 검증 데이터 예측
    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    # 테스트 데이터 예측 누적
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits
    
    # 점수 계산
    score = roc_auc_score(y_va, val_pred)
    fold_scores.append(score)
    print(f"Fold {fold+1} ROC-AUC: {score:.5f}")

mean_auc = np.mean(fold_scores)
print(f"\nMean ROC-AUC Score (OOF): {mean_auc:.5f}")

# 5. 제출 파일 저장
submission['임신 성공 여부'] = test_preds
output_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v1_lgb.csv")
submission.to_csv(output_sub_path, index=False)
print(f"Saved LightGBM submission to: {output_sub_path}")
