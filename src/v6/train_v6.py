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

# 3. v5 정밀 결측치 대치용 Train 통계치
train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0

# 4. v6 이상치 제어용 Train 99.5% 백분위값 사전 산출
clip_cols = ['수집된 신선 난자 수', '저장된 신선 난자 수', '총 생성 배아 수', '이식된 배아 수', '저장된 배아 수']
clip_limits = {}
for col in clip_cols:
    limit = train[col].quantile(0.995)
    clip_limits[col] = limit
    print(f"Outlier Limit (99.5%) for '{col}': {limit}")

# 5. v6 Groupby Aggregation용 통계 맵 사전 구축 (Train 기준)
count_map = {
    '0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6
}
temp_train = train.copy()
temp_train['총 출산 횟수_int'] = temp_train['총 출산 횟수'].map(count_map).fillna(0)

age_egg_mean = train.groupby('시술 당시 나이')['수집된 신선 난자 수'].mean().to_dict()
age_egg_std = train.groupby('시술 당시 나이')['수집된 신선 난자 수'].std().to_dict()
age_embryo_mean = train.groupby('시술 당시 나이')['총 생성 배아 수'].mean().to_dict()
age_embryo_std = train.groupby('시술 당시 나이')['총 생성 배아 수'].std().to_dict()
cause_delivery_mean = temp_train.groupby('여성 주 불임 원인')['총 출산 횟수_int'].mean().to_dict()

# 6. 도메인 피처 엔지니어링 및 전처리 함수
def preprocess_data(df, transfer_median):
    df_new = df.copy()
    
    # 6.1 [v5 정밀 결측 대치] 배아 이식 경과일
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median
    df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)
    
    # 6.2 [v5 정밀 결측 대치] 난자/정자 기증자 나이 진짜 결측치 최빈값 대치
    df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
    df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'
    
    # 6.3 [v6 이상치 제어] 99.5% 백분위 클리핑 (상한선 제한)
    for col in clip_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].clip(upper=clip_limits[col])
            
    # 6.4 [v6 Groupby Aggregation] 통계 피처 매핑
    df_new['age_egg_mean'] = df_new['시술 당시 나이'].map(age_egg_mean).fillna(0)
    df_new['age_egg_std'] = df_new['시술 당시 나이'].map(age_egg_std).fillna(0)
    df_new['age_embryo_mean'] = df_new['시술 당시 나이'].map(age_embryo_mean).fillna(0)
    df_new['age_embryo_std'] = df_new['시술 당시 나이'].map(age_embryo_std).fillna(0)
    df_new['cause_delivery_mean'] = df_new['여성 주 불임 원인'].map(cause_delivery_mean).fillna(0)
    
    # 6.5 나이 Ordinal Encoding
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
    
    # 6.6 고령 임산부 기준 플래그 생성
    df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(int)
    df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(int)
    
    # 6.7 시술/임신/출산 횟수 문자열 -> 정수형 변환
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        df_new[f'{col}_int'] = df_new[col].map(count_map).fillna(0)
        
    # 6.8 파생 피처: 이전 성공률
    df_new['pregnancy_efficiency'] = df_new['총 임신 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    df_new['delivery_efficiency'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    df_new['previous_success_rate'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    
    # 6.9 [v6 교차 피처 고도화] 시술 횟수 대비 난수 생산비율
    df_new['egg_per_cycle'] = df_new['수집된 신선 난자 수'] / (df_new['총 시술 횟수_int'] + 1)
    
    # 6.10 파생 피처: 고령 임산부 및 기증 난자 상호작용 피처
    df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(int)
    df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(int)
    df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(int)
    
    # 6.11 파생 피처: 이식 미실시 및 동결 전용 주기 플래그
    df_new['is_transfer_missing'] = ((df_new['배아 이식 경과일'].isna()) | (df_new['배아 이식 경과일'] == -1) | (df_new['이식된 배아 수'] == 0)).astype(int)
    df_new['frozen_only_cycle'] = (((df_new['배아 이식 경과일'].isna()) | (df_new['배아 이식 경과일'] == -1)) & (df_new['저장된 배아 수'] > 0)).astype(int)
    
    # 6.12 파생 피처: 배아 배양 일수 계산
    df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']
    df_new['embryo_culture_days'] = df_new['embryo_culture_days'].fillna(-1)
    
    # 6.13 파생 피처: 난수/배아 관련 비율 지표
    df_new['egg_to_embryo_ratio'] = df_new['총 생성 배아 수'] / (df_new['수집된 신선 난자 수'] + 1)
    df_new['egg_to_embryo_ratio'] = df_new['egg_to_embryo_ratio'].fillna(0)
    df_new['embryo_stored_ratio'] = df_new['저장된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
    df_new['embryo_transferred_ratio'] = df_new['이식된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
    
    # 6.14 시술 특성별 결측치 조건부 처리
    fresh_egg_cols = ['수집된 신선 난자 수', '저장된 신선 난자 수']
    for col in fresh_egg_cols:
        df_new.loc[df_new['동결 배아 사용 여부'] == 1, col] = df_new.loc[df_new['동결 배아 사용 여부'] == 1, col].fillna(-1)
        
    thaw_cols = ['해동된 배아 수', '해동 난자 수', '난자 해동 경과일', '배아 해동 경과일']
    for col in thaw_cols:
        df_new.loc[df_new['신선 배아 사용 여부'] == 1, col] = df_new.loc[df_new['신선 배아 사용 여부'] == 1, col].fillna(-1)
        
    # 6.15 남성/여성 불임 요인 통합 플래그 생성
    infertility_factor_cols = [
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', 
        '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', 
        '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'
    ]
    df_new['infertility_severity_score'] = df_new[infertility_factor_cols].sum(axis=1)
    
    male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
    df_new['is_male_infertility'] = df_new[male_factors].any(axis=1).astype(int)
    
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
        
        X[col] = pd.Categorical(X[col], categories=full_categories)
        X_test[col] = pd.Categorical(X_test[col], categories=full_categories)
        cat_cols.append(col)

print(f"Total features after v6 additions: {len(features)}")
print(f"Categorical features: {len(cat_cols)}")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================== 7. Model 1: LightGBM ====================
print("\n=== Training Model 1: LightGBM ===")
lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))

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
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
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
    
    lgb_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    lgb_test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

lgb_score = roc_auc_score(y, lgb_oof)
print(f"LightGBM OOF ROC-AUC: {lgb_score:.5f}")


# ==================== 8. Model 2: XGBoost ====================
print("\n=== Training Model 2: XGBoost ===")
xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=1.5,
        tree_method='hist',
        enable_categorical=True,
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
    
    xgb_oof[val_idx] = model.predict_proba(X_va)[:, 1]
    xgb_test += model.predict_proba(X_test)[:, 1] / folds.n_splits

xgb_score = roc_auc_score(y, xgb_oof)
print(f"XGBoost OOF ROC-AUC: {xgb_score:.5f}")


# ==================== 9. Model 3: CatBoost ====================
print("\n=== Training Model 3: CatBoost ===")
cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))

X_cat = X.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    X_cat[col] = X_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(folds.split(X_cat, y)):
    X_tr, y_tr = X_cat.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_cat.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=1500,
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
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    
    cat_oof[val_idx] = model.predict_proba(X_va)[:, 1]
    cat_test += model.predict_proba(X_test_cat)[:, 1] / folds.n_splits

cat_score = roc_auc_score(y, cat_oof)
print(f"CatBoost OOF ROC-AUC: {cat_score:.5f}")


# ==================== 10. Weighted Ensemble ====================
print("\n=== Calculating Weighted Ensemble ===")

w_lgb, w_xgb, w_cat = 0.40, 0.30, 0.30

ensemble_oof = (w_lgb * lgb_oof) + (w_xgb * xgb_oof) + (w_cat * cat_oof)
ensemble_score = roc_auc_score(y, ensemble_oof)

print(f"LightGBM Weight: {w_lgb:.2f} | XGBoost Weight: {w_xgb:.2f} | CatBoost Weight: {w_cat:.2f}")
print(f"Ensemble OOF ROC-AUC Score: {ensemble_score:.5f}")

# 11. 최종 제출 파일 저장
final_test_preds = (w_lgb * lgb_test) + (w_xgb * xgb_test) + (w_cat * cat_test)
submission['임신 성공 여부'] = final_test_preds

output_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v6_advanced.csv")
submission.to_csv(output_sub_path, index=False)
print(f"Saved Ensemble submission to: {output_sub_path}")
