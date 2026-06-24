import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import optuna

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Optuna 로그 숨김 설정 (오류만 출력)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# 3. 진짜 결측 대치용 Train 통계치 산정
train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0  # 폴백

print(f"Calculated Train Actual Transfer Median Days: {train_transfer_median}")

# 4. 도메인 피처 엔지니어링 및 전처리 함수 (v11 - 하드케이스 억제 및 범주 코드 변환 포함)
def preprocess_data(df, transfer_median):
    df_new = df.copy()
    
    # 4.1 [시술 유형 정보 역산 보간]
    df_new.loc[(df_new['특정 시술 유형'] == 'Unknown') & (df_new['미세주입된 난자 수'] > 0), '특정 시술 유형'] = 'ICSI'
    
    # 4.2 [결측 인디케이터 설계]
    df_new['is_missing_eggs'] = df_new['수집된 신선 난자 수'].isna().astype(int)
    df_new['is_missing_embryos'] = df_new['총 생성 배아 수'].isna().astype(int)
    df_new['is_missing_transfer'] = df_new['배아 이식 경과일'].isna().astype(int)
    
    # 4.3 배아 이식 경과일 진짜/가짜 결측 분리 대치
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median
    df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)
    
    # 4.4 난자 기증자 나이 결측치 대치
    df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
    
    # 4.5 정자 기증자 나이 결측치 대치
    df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'
    
    # 4.6 나이 Ordinal Encoding
    age_map = {
        '만18-34세': 0, '만35-37세': 1, '만38-39세': 2, 
        '만40-42세': 3, '만43-44세': 4, '만45-50세': 5, '알 수 없음': -1
    }
    donor_age_map = {
        '만20세 이하': 0, '만21-25세': 1, '만26-30세': 2, 
        '만31-35세': 3, '만36-40세': 4, '만41-45세': 5, '알 수 없음': -1
    }
    
    df_new['시술 당시 나이_ordinal'] = df_new['시술 당시 나이'].map(age_map).fillna(-1)
    df_new['난자 기증자 나이_ordinal'] = df_new['난자 기증자 나이'].map(donor_age_map).fillna(-1)
    df_new['정자 기증자 나이_ordinal'] = df_new['정자 기증자 나이'].map(donor_age_map).fillna(-1)
    
    # 4.7 고령 임산부 기준 플래그
    df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(int)
    df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(int)
    
    # 4.8 DI 시술군 특화 가임력 상호작용 피처
    df_new['is_DI_cycle'] = (df_new['시술 유형'] == 'DI').astype(int)
    df_new['DI_young_age'] = ((df_new['is_DI_cycle'] == 1) & (df_new['시술 당시 나이_ordinal'] <= 1)).astype(int)
    df_new['DI_advanced_age'] = ((df_new['is_DI_cycle'] == 1) & (df_new['시술 당시 나이_ordinal'] >= 2)).astype(int)
    
    # 4.9 시술/임신/출산 횟수 문자열 -> 정수형 변환
    count_map = {
        '0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6
    }
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        df_new[f'{col}_int'] = df_new[col].map(count_map).fillna(0)
        
    # 4.10 파생 피처: 이전 성공률
    df_new['pregnancy_efficiency'] = df_new['총 임신 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    df_new['delivery_efficiency'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    df_new['previous_success_rate'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    
    # 4.11 파생 피처: 고령 임산부 및 기증 난자 상호작용 피처
    df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(int)
    df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(int)
    df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(int)
    
    # 4.12 파생 피처: 이식 미실시 및 동결 전용 주기 플래그
    df_new['is_transfer_missing'] = ((df_new['배아 이식 경과일'].isna()) | (df_new['배아 이식 경과일'] == -1) | (df_new['이식된 배아 수'] == 0)).astype(int)
    df_new['frozen_only_cycle'] = (((df_new['배아 이식 경과일'].isna()) | (df_new['배아 이식 경과일'] == -1)) & (df_new['저장된 배아 수'] > 0)).astype(int)
    
    # 4.13 파생 피처: 배아 배양 일수 계산
    df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']
    df_new['embryo_culture_days'] = df_new['embryo_culture_days'].fillna(-1)
    
    # 4.14 파생 피처: 난수/배아 관련 비율 지표
    df_new['egg_to_embryo_ratio'] = df_new['총 생성 배아 수'] / (df_new['수집된 신선 난자 수'] + 1)
    df_new['egg_to_embryo_ratio'] = df_new['egg_to_embryo_ratio'].fillna(0)
    df_new['embryo_stored_ratio'] = df_new['저장된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
    df_new['embryo_transferred_ratio'] = df_new['이식된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
    
    # [거짓 음성(FN) 방어 피처 설계]
    df_new['embryo_transfer_to_collected_ratio'] = df_new['이식된 배아 수'] / (df_new['수집된 신선 난자 수'] + 1)
    df_new['embryo_transfer_to_collected_ratio'] = df_new['embryo_transfer_to_collected_ratio'].fillna(0)
    
    df_new['thaw_survival_rate'] = df_new['해동된 배아 수'] / (df_new['해동 난자 수'] + 1)
    df_new['thaw_survival_rate'] = df_new['thaw_survival_rate'].fillna(0)
    
    df_new['FET_young_age'] = ((df_new['동결 배아 사용 여부'] == 1) & (df_new['시술 당시 나이_ordinal'] <= 1)).astype(int)
    
    # [거짓 양성(FP) 방어 피처 설계]
    df_new['ohss_high_risk_flag'] = (
        ((df_new['수집된 신선 난자 수'] >= 15.0) | (df_new['총 생성 배아 수'] >= 8.0)) & 
        (df_new['신선 배아 사용 여부'] == 1)
    ).astype(int)
    
    infertility_factor_cols = [
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', 
        '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', 
        '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'
    ]
    df_new['infertility_severity_score'] = df_new[infertility_factor_cols].sum(axis=1)
    male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
    df_new['is_male_infertility'] = df_new[male_factors].any(axis=1).astype(int)
    
    df_new['male_factor_severity_interact'] = df_new['infertility_severity_score'] * df_new['is_male_infertility']
    
    # 4.15 시술 특성별 결측치 조건부 처리
    fresh_egg_cols = ['수집된 신선 난자 수', '저장된 신선 난자 수']
    for col in fresh_egg_cols:
        df_new.loc[df_new['동결 배아 사용 여부'] == 1, col] = df_new.loc[df_new['동결 배아 사용 여부'] == 1, col].fillna(-1)
        
    thaw_cols = ['해동된 배아 수', '해동 난자 수', '난자 해동 경과일', '배아 해동 경과일']
    for col in thaw_cols:
        df_new.loc[df_new['신선 배아 사용 여부'] == 1, col] = df_new.loc[df_new['신선 배아 사용 여부'] == 1, col].fillna(-1)
        
    female_factors = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '여성 주 불임 원인', '여성 부 불임 원인']
    df_new['is_female_infertility'] = df_new[female_factors].any(axis=1).astype(int)

    # 기존 원본 컬럼 중 수치화된 컬럼 제거
    cols_to_drop = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + count_cols
    df_new = df_new.drop(columns=cols_to_drop)
    
    return df_new

# 전처리 수행
print("Preprocessing datasets...")
X = preprocess_data(train, train_transfer_median)
X_test = preprocess_data(test, train_transfer_median)

# 피처 리스트 구성
drop_cols = ['ID', target_col] if target_col in X.columns else ['ID']
features = [col for col in X.columns if col not in drop_cols]
y = train[target_col]

X = X[features].copy()
X_test = X_test[features].copy()

# 범주형 컬럼 변환 및 카테고리 일치
cat_cols = []
for col in features:
    if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
        X[col] = X[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)
        
        full_categories = sorted(list(set(X[col].unique().tolist() + X_test[col].unique().tolist())))
        
        X[col] = pd.Categorical(X[col], categories=full_categories)
        X_test[col] = pd.Categorical(X_test[col], categories=full_categories)
        
        # 카테고리 코드 변환 (학습 속도 최적화)
        X[col] = X[col].cat.codes
        X_test[col] = X_test[col].cat.codes
        cat_cols.append(col)

print(f"Total features: {len(features)}")
print(f"Categorical features: {len(cat_cols)}")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 시술 유형 인덱스 분리
ivf_mask_train = (train['시술 유형'] == 'IVF').values
di_mask_train = (train['시술 유형'] == 'DI').values
ivf_mask_test = (test['시술 유형'] == 'IVF').values
di_mask_test = (test['시술 유형'] == 'DI').values


# ==================== 5. Hardcoded Hyperparameters from Optuna ====================
print("\n=== Applying Best Parameters from Optuna ===")
best_lgb_params = {
    'num_leaves': 84,
    'max_depth': 5,
    'scale_pos_weight': 1.1010674815175636,
    'feature_fraction': 0.75820073990071,
    'bagging_fraction': 0.828389374170551
}
best_xgb_params = {
    'max_depth': 4,
    'subsample': 0.7294098601988732,
    'colsample_bytree': 0.505053680407901,
    'scale_pos_weight': 1.9239872813343522
}
print(f"Loaded LGBM parameters: {best_lgb_params}")
print(f"Loaded XGBoost parameters: {best_xgb_params}")


# ==================== 6. Base Model Training on Full Dataset ====================
print("\n=== Training Base Models on Full Dataset ===")

# 6.1 Tuned LightGBM
print("\n--- Training Tuned Joint LightGBM ---")
joint_lgb_oof = np.zeros(len(X))
joint_lgb_test = np.zeros(len(X_test))

lgb_final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02, # 최적화를 위해 보수적 학습률 적용
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
lgb_final_params.update(best_lgb_params)

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    model = lgb.train(
        lgb_final_params, trn_data, num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    joint_lgb_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    joint_lgb_test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

lgb_score = roc_auc_score(y, joint_lgb_oof)
print(f"Tuned Joint LGB OOF AUC: {lgb_score:.6f}")

# 6.2 Tuned XGBoost
print("\n--- Training Tuned Joint XGBoost ---")
joint_xgb_oof = np.zeros(len(X))
joint_xgb_test = np.zeros(len(X_test))

xgb_final_params = {
    'n_estimators': 1500,
    'learning_rate': 0.02,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'early_stopping_rounds': 50
}
xgb_final_params.update(best_xgb_params)

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**xgb_final_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    joint_xgb_oof[val_idx] = model.predict_proba(X_va)[:, 1]
    joint_xgb_test += model.predict_proba(X_test)[:, 1] / folds.n_splits

xgb_score = roc_auc_score(y, joint_xgb_oof)
print(f"Tuned Joint XGB OOF AUC: {xgb_score:.6f}")

# 6.3 Joint CatBoost (v10 optimized)
print("\n--- Training Joint CatBoost ---")
joint_cat_oof = np.zeros(len(X))
joint_cat_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        task_type='CPU',
        early_stopping_rounds=50,
        verbose=False
    )
    
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    
    joint_cat_oof[val_idx] = model.predict_proba(X_va)[:, 1]
    joint_cat_test += model.predict_proba(X_test)[:, 1] / folds.n_splits

cat_score = roc_auc_score(y, joint_cat_oof)
print(f"Joint CatBoost OOF AUC: {cat_score:.6f}")

# 6.4 Split IVF LightGBM
print("\n--- Training Split Model 1: IVF Specific LightGBM ---")
ivf_oof = np.zeros(sum(ivf_mask_train))
ivf_test = np.zeros(sum(ivf_mask_test))

X_ivf = X[ivf_mask_train].copy()
y_ivf = y[ivf_mask_train]
X_test_ivf = X_test[ivf_mask_test].copy()
folds_ivf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(folds_ivf.split(X_ivf, y_ivf)):
    X_tr, y_tr = X_ivf.iloc[train_idx], y_ivf.iloc[train_idx]
    X_va, y_va = X_ivf.iloc[val_idx], y_ivf.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_va, label=y_va)
    
    model = lgb.train(
        lgb_final_params, trn_data, num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    ivf_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    ivf_test += model.predict(X_test_ivf, num_iteration=model.best_iteration) / folds_ivf.n_splits

# 6.5 Split DI LightGBM
print("\n--- Training Split Model 2: DI Specific LightGBM ---")
di_oof = np.zeros(sum(di_mask_train))
di_test = np.zeros(sum(di_mask_test))

X_di = X[di_mask_train].copy()
y_di = y[di_mask_train]
X_test_di = X_test[di_mask_test].copy()

lgb_params_di = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 15,
    'max_depth': 4,
    'scale_pos_weight': 1.5,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
folds_di = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(folds_di.split(X_di, y_di)):
    X_tr, y_tr = X_di.iloc[train_idx], y_di.iloc[train_idx]
    X_va, y_va = X_di.iloc[val_idx], y_di.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_va, label=y_va)
    
    model = lgb.train(
        lgb_params_di, trn_data, num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    di_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    di_test += model.predict(X_test_di, num_iteration=model.best_iteration) / folds_di.n_splits


# ==================== 7. Meta Stacking Model (L2 Logistic Regression) ====================
print("\n=== Training Meta Stacking Model (L2 Logistic Regression) ===")

# consolidated Split OOF 생성
split_oof = np.zeros(len(X))
split_oof[ivf_mask_train] = ivf_oof
split_oof[di_mask_train] = di_oof

split_test = np.zeros(len(X_test))
split_test[ivf_mask_test] = ivf_test
split_test[di_mask_test] = di_test

# 메타 학습 피처 행렬 구축 (4개 차원)
X_meta = np.column_stack([
    joint_lgb_oof,
    joint_xgb_oof,
    joint_cat_oof,
    split_oof
])

X_meta_test = np.column_stack([
    joint_lgb_test,
    joint_xgb_test,
    joint_cat_test,
    split_test
])

meta_oof = np.zeros(len(X))
meta_test = np.zeros(len(X_test))

# 5-fold CV 학습 (메타 과적합 방지)
for fold, (train_idx, val_idx) in enumerate(folds.split(X_meta, y)):
    X_tr, y_tr = X_meta[train_idx], y.iloc[train_idx]
    X_va, y_va = X_meta[val_idx], y.iloc[val_idx]
    
    meta_model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', random_state=42)
    meta_model.fit(X_tr, y_tr)
    
    meta_oof[val_idx] = meta_model.predict_proba(X_va)[:, 1]
    meta_test += meta_model.predict_proba(X_meta_test)[:, 1] / folds.n_splits

# 최종 메타 모델 가중치 분석용 학습
final_meta = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', random_state=42)
final_meta.fit(X_meta, y)

print("\n--- Meta Stacking Weights Analysis ---")
print(f"Weight for Joint LGB: {final_meta.coef_[0][0]:.4f}")
print(f"Weight for Joint XGB: {final_meta.coef_[0][1]:.4f}")
print(f"Weight for Joint Cat: {final_meta.coef_[0][2]:.4f}")
print(f"Weight for Split LGB (Domain): {final_meta.coef_[0][3]:.4f}")
print(f"Meta Bias (Intercept): {final_meta.intercept_[0]:.4f}")


# ==================== 8. Performance Evaluation & Reports ====================
print("\n=== Model Performance Comparison ===")
print(f"1) Tuned Joint LightGBM OOF AUC: {lgb_score:.6f}")
print(f"2) Tuned Joint XGBoost OOF AUC: {xgb_score:.6f}")
print(f"3) Joint CatBoost OOF AUC: {cat_score:.6f}")

# Joint Ensemble 스코어 산출 (v5/v10 가중 평균 비교용)
w_lgb, w_xgb, w_cat = 0.40, 0.30, 0.30
joint_ens_oof = (w_lgb * joint_lgb_oof) + (w_xgb * joint_xgb_oof) + (w_cat * joint_cat_oof)
joint_ens_test = (w_lgb * joint_lgb_test) + (w_xgb * joint_xgb_test) + (w_cat * joint_cat_test)
joint_ens_score = roc_auc_score(y, joint_ens_oof)
print(f"4) Simple Joint Ensemble OOF AUC: {joint_ens_score:.6f}")

# 최종 v11 Stacking 스코어
stacking_score = roc_auc_score(y, meta_oof)
print(f"\n5) v11 Meta Stacking OOF AUC: {stacking_score:.6f}")
print(f"   - Stacking IVF subset AUC: {roc_auc_score(y[ivf_mask_train], meta_oof[ivf_mask_train]):.6f}")
print(f"   - Stacking DI subset AUC: {roc_auc_score(y[di_mask_train], meta_oof[di_mask_train]):.6f}")


# ==================== 9. Save Submission ====================
# 성능이 더 뛰어난 Tuned Joint Ensemble 예측 저장
submission['probability'] = joint_ens_test
output_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v11_advanced.csv")
submission.to_csv(output_sub_path, index=False)
print(f"\nSaved v11 Tuned Joint Ensemble submission to: {output_sub_path}")

# 스태킹 결과도 백업용으로 저장
submission['probability'] = meta_test
backup_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v11_stacking.csv")
submission.to_csv(backup_sub_path, index=False)
print(f"Saved v11 Stacking submission to: {backup_sub_path}")
