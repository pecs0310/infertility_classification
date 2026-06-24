import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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
    train_transfer_median = 3.0  # 폴백 (임상상 가장 흔한 3일 배아 이식 기준)

print(f"Calculated Train Actual Transfer Median Days: {train_transfer_median}")

# 4. 도메인 피처 엔지니어링 및 전처리 함수 (v10 - 하드케이스 억제 및 앙상블 전처리)
def preprocess_data(df, transfer_median):
    df_new = df.copy()
    
    # 4.1 [시술 유형 정보 역산 보간]
    df_new.loc[(df_new['특정 시술 유형'] == 'Unknown') & (df_new['미세주입된 난자 수'] > 0), '특정 시술 유형'] = 'ICSI'
    
    # 4.2 [결측 인디케이터 설계] 결측 여부 자체를 피처로 보존
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
        
        # Pandas Categorical로 변환
        X[col] = pd.Categorical(X[col], categories=full_categories)
        X_test[col] = pd.Categorical(X_test[col], categories=full_categories)
        
        # XGBoost와 CatBoost의 속도 향상을 위해 정수형 코드(cat.codes)로 변환
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

# ==================== 1. Joint LightGBM Model ====================
print("\n=== Training Joint Model 1: LightGBM ===")
joint_lgb_oof = np.zeros(len(X))
joint_lgb_test = np.zeros(len(X_test))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 9,
    'scale_pos_weight': 1.5,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM은 정수 인코딩된 범주형 피처 지정 가능
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks
    )
    
    joint_lgb_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    joint_lgb_test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

lgb_score = roc_auc_score(y, joint_lgb_oof)
print(f"Joint LightGBM OOF ROC-AUC: {lgb_score:.5f}")


# ==================== 2. Joint XGBoost Model ====================
print("\n=== Training Joint Model 2: XGBoost ===")
joint_xgb_oof = np.zeros(len(X))
joint_xgb_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # 카테고리가 정수 코드로 변환되었으므로 일반 numerical 데이터처럼 초고속 학습 가능
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=1.5,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    
    joint_xgb_oof[val_idx] = model.predict_proba(X_va)[:, 1]
    joint_xgb_test += model.predict_proba(X_test)[:, 1] / folds.n_splits

xgb_score = roc_auc_score(y, joint_xgb_oof)
print(f"Joint XGBoost OOF ROC-AUC: {xgb_score:.5f}")


# ==================== 3. Joint CatBoost Model ====================
print("\n=== Training Joint Model 3: CatBoost ===")
joint_cat_oof = np.zeros(len(X))
joint_cat_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # cat_features 지정을 없앰으로써 100배 빠른 CPU 수치 연산으로 학습
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
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    
    joint_cat_oof[val_idx] = model.predict_proba(X_va)[:, 1]
    joint_cat_test += model.predict_proba(X_test)[:, 1] / folds.n_splits

cat_score = roc_auc_score(y, joint_cat_oof)
print(f"Joint CatBoost OOF ROC-AUC: {cat_score:.5f}")


# ==================== 4. Joint Ensemble (LGB 40% + XGB 30% + Cat 30%) ====================
print("\n=== Calculating Joint Ensemble Predictions ===")
w_lgb, w_xgb, w_cat = 0.40, 0.30, 0.30

joint_ens_oof = (w_lgb * joint_lgb_oof) + (w_xgb * joint_xgb_oof) + (w_cat * joint_cat_oof)
joint_ens_test = (w_lgb * joint_lgb_test) + (w_xgb * joint_xgb_test) + (w_cat * joint_cat_test)

joint_ens_score = roc_auc_score(y, joint_ens_oof)
print(f"Joint Ensemble OOF ROC-AUC: {joint_ens_score:.5f}")


# ==================== 5. IVF Specific LightGBM Model (Split IVF) ====================
print("\n=== Training Split Model 1: IVF Specific LightGBM ===")
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
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    ivf_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    ivf_test += model.predict(X_test_ivf, num_iteration=model.best_iteration) / folds_ivf.n_splits


# ==================== 6. DI Specific LightGBM Model (Split DI) ====================
print("\n=== Training Split Model 2: DI Specific LightGBM ===")
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
        lgb_params_di,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    di_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    di_test += model.predict(X_test_di, num_iteration=model.best_iteration) / folds_di.n_splits


# ==================== 7. Conditional Blending & Evaluation ====================
print("\n=== Performing Conditional Blending (Joint Ensemble + Split Models) ===")

# 조건부 가중치 블렌딩 (IVF는 Joint 50% + IVF전용 50%, DI는 Joint 30% + DI전용 70%)
lambda_ivf = 0.50
lambda_di = 0.30

final_oof = np.zeros(len(X))
final_oof[ivf_mask_train] = (lambda_ivf * joint_ens_oof[ivf_mask_train]) + ((1 - lambda_ivf) * ivf_oof)
final_oof[di_mask_train] = (lambda_di * joint_ens_oof[di_mask_train]) + ((1 - lambda_di) * di_oof)

final_test = np.zeros(len(X_test))
final_test[ivf_mask_test] = (lambda_ivf * joint_ens_test[ivf_mask_test]) + ((1 - lambda_ivf) * ivf_test)
final_test[di_mask_test] = (lambda_di * joint_ens_test[di_mask_test]) + ((1 - lambda_di) * di_test)

# 평가 리포트
print("\n--- Model Performance Comparison ---")
print(f"1) Joint LightGBM OOF AUC: {lgb_score:.6f}")
print(f"2) Joint XGBoost OOF AUC: {xgb_score:.6f}")
print(f"3) Joint CatBoost OOF AUC: {cat_score:.6f}")
print(f"4) Joint Ensemble OOF AUC: {joint_ens_score:.6f}")
print(f"   - Joint Ens IVF subset AUC: {roc_auc_score(y[ivf_mask_train], joint_ens_oof[ivf_mask_train]):.6f}")
print(f"   - Joint Ens DI subset AUC: {roc_auc_score(y[di_mask_train], joint_ens_oof[di_mask_train]):.6f}")

final_score = roc_auc_score(y, final_oof)
print(f"\n5) v10 Conditional Blending (Joint Ensemble + Split LGB) OOF AUC: {final_score:.6f}")
print(f"   - Blended IVF subset AUC: {roc_auc_score(y[ivf_mask_train], final_oof[ivf_mask_train]):.6f}")
print(f"   - Blended DI subset AUC: {roc_auc_score(y[di_mask_train], final_oof[di_mask_train]):.6f}")

# 8. 최종 제출 파일 저장
# 성능이 더 뛰어난 Joint Ensemble 예측 저장
submission['probability'] = joint_ens_test
output_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v10_advanced.csv")
submission.to_csv(output_sub_path, index=False)
print(f"\nSaved v10 Joint Ensemble submission to: {output_sub_path}")

# 조건부 블렌딩 결과도 백업용으로 저장
submission['probability'] = final_test
backup_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v10_blended.csv")
submission.to_csv(backup_sub_path, index=False)
print(f"Saved v10 Blended submission to: {backup_sub_path}")
