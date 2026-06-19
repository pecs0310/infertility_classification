import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. 경로 설정
# 실행 위치에 상관없이 데이터 경로를 절대 경로로 잡거나 상대 경로로 유연하게 대처합니다.
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

# 3. 전처리 (간단한 결측치 처리 및 Label Encoding)
X = train[features].copy()
y = train[target_col]
X_test = test[features].copy()

# 범주형 컬럼과 수치형 컬럼 구분
cat_cols = []
num_cols = []

for col in features:
    if X[col].dtype == 'object':
        cat_cols.append(col)
    else:
        num_cols.append(col)

print(f"Categorical features: {len(cat_cols)}")
print(f"Numerical features: {len(num_cols)}")

# 결측치 채우기
for col in num_cols:
    mean_val = X[col].mean() if not pd.isna(X[col].mean()) else 0
    X[col] = X[col].fillna(mean_val)
    X_test[col] = X_test[col].fillna(mean_val)

for col in cat_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

# Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    # train과 test 데이터셋 전체를 커버할 수 있도록 fit
    full_unique = pd.concat([X[col], X_test[col]]).unique()
    le.fit(full_unique)
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

# 4. K-Fold Validation (ROC-AUC)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\n--- Starting K-Fold Validation (Random Forest Baseline) ---")
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    
    # 검증 데이터 예측 (임신 성공 확률: class 1 probability)
    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred
    
    # 테스트 데이터 예측 누적
    test_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
    
    # 점수 계산
    score = roc_auc_score(y_va, val_pred)
    fold_scores.append(score)
    print(f"Fold {fold+1} ROC-AUC: {score:.5f}")

mean_auc = np.mean(fold_scores)
print(f"\nMean ROC-AUC Score: {mean_auc:.5f}")

# 5. 제출 파일 저장
submission['임신 성공 여부'] = test_preds
output_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v1.csv")
submission.to_csv(output_sub_path, index=False)
print(f"Saved baseline submission to: {output_sub_path}")
