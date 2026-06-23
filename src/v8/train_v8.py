import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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

# 4. 도메인 피처 엔지니어링 및 전처리 함수 (v8 고도화 버전)
def preprocess_data(df, transfer_median):
    df_new = df.copy()
    
    # 4.1 배아 이식 경과일 진짜/가짜 결측 분리 대치
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median
    df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)
    
    # 4.2 난자 기증자 나이 결측치 대치
    df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
    
    # 4.3 정자 기증자 나이 결측치 대치
    df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'
    
    # 4.4 나이 Ordinal Encoding
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
    
    # 4.5 고령 임산부 기준 플래그
    df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(int)
    df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(int)
    
    # 4.6 시술/임신/출산 횟수 문자열 -> 정수형 변환
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
        
    # 4.7 파생 피처: 이전 성공률
    df_new['pregnancy_efficiency'] = df_new['총 임신 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    df_new['delivery_efficiency'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    df_new['previous_success_rate'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
    
    # 4.8 파생 피처: 고령 임산부 및 기증 난자 상호작용 피처
    df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(int)
    df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(int)
    df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(int)
    
    # [v8 신규 고도화 피처 1] 고령 환자의 젊은 난자 공여 시너지 매칭 플래그
    # 만 40세 이상(ordinal >= 3) 환자가 젊은 기증자(만 30세 이하, donor ordinal <= 2) 난자 공여를 적극 수혜받은 고예후 시너지 군집 식별
    df_new['elderly_donor_mismatch_synergy'] = (
        (df_new['is_elderly_age'] == 1) & 
        (df_new['난자 출처'] == '기증 제공') & 
        (df_new['난자 기증자 나이_ordinal'] >= 0) & 
        (df_new['난자 기증자 나이_ordinal'] <= 2)
    ).astype(int)
    
    # 4.9 파생 피처: 이식 미실시 및 동결 전용 주기 플래그
    df_new['is_transfer_missing'] = ((df_new['배아 이식 경과일'].isna()) | (df_new['배아 이식 경과일'] == -1) | (df_new['이식된 배아 수'] == 0)).astype(int)
    df_new['frozen_only_cycle'] = (((df_new['배아 이식 경과일'].isna()) | (df_new['배아 이식 경과일'] == -1)) & (df_new['저장된 배아 수'] > 0)).astype(int)
    
    # 4.10 파생 피처: 배아 배양 일수 계산
    df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']
    df_new['embryo_culture_days'] = df_new['embryo_culture_days'].fillna(-1)
    
    # 4.11 파생 피처: 난수/배아 관련 비율 지표
    df_new['egg_to_embryo_ratio'] = df_new['총 생성 배아 수'] / (df_new['수집된 신선 난자 수'] + 1)
    df_new['egg_to_embryo_ratio'] = df_new['egg_to_embryo_ratio'].fillna(0)
    df_new['embryo_stored_ratio'] = df_new['저장된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
    df_new['embryo_transferred_ratio'] = df_new['이식된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
    
    # [v8 신규 고도화 피처 2] 난자 및 배아 생성 효율 추가 다각화
    # - 수정률(ICSI Fertilization Rate): 미세주입 난자 대비 생성된 배아의 비율
    df_new['icsi_fertilization_rate'] = df_new['미세주입에서 생성된 배아 수'] / (df_new['미세주입된 난자 수'] + 1)
    df_new['icsi_fertilization_rate'] = df_new['icsi_fertilization_rate'].fillna(0)
    
    # - 배아 이식 효율(Embryo Transfer Efficiency): 저장 대비 이식된 배아 효율
    df_new['embryo_transfer_efficiency'] = df_new['이식된 배아 수'] / (df_new['저장된 배아 수'] + 1)
    df_new['embryo_transfer_efficiency'] = df_new['embryo_transfer_efficiency'].fillna(0)
    
    # 4.12 시술 특성별 결측치 조건부 처리
    fresh_egg_cols = ['수집된 신선 난자 수', '저장된 신선 난자 수']
    for col in fresh_egg_cols:
        df_new.loc[df_new['동결 배아 사용 여부'] == 1, col] = df_new.loc[df_new['동결 배아 사용 여부'] == 1, col].fillna(-1)
        
    thaw_cols = ['해동된 배아 수', '해동 난자 수', '난자 해동 경과일', '배아 해동 경과일']
    for col in thaw_cols:
        df_new.loc[df_new['신선 배아 사용 여부'] == 1, col] = df_new.loc[df_new['신선 배아 사용 여부'] == 1, col].fillna(-1)
        
    # 4.13 남성/여성 불임 요인 통합 플래그 생성
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

    # [v8 신규 고도화 피처 3] 초고위험 / 고예후 특정 시술 조합 바이너리 플래그 생성
    # - 초고위험군 특정 시술 유형 (성공률이 1% 이하인 ICSI:ICSI, IVF:IVF 등 예후가 극도로 좋지 않은 군)
    df_new['is_extreme_low_risk_procedure'] = df_new['특정 시술 유형'].isin(['ICSI:ICSI', 'IVF:IVF']).astype(int)
    # - 고예후 특정 시술 유형 (성공률이 35% 이상으로 높은 ICSI / BLASTOCYST, IVF / BLASTOCYST 등)
    df_new['is_high_prognosis_procedure'] = df_new['특정 시술 유형'].isin(['ICSI / BLASTOCYST', 'IVF / BLASTOCYST']).astype(int)

    # [v8 신규 고도화 피처 4] PGS / PGD 유전 검사 여부 정보 보존 플래그
    # PGS/PGD 검사는 결측률이 매우 높지만, 1.0인 경우 기저 난임의 수준을 대변하는 저예후 임상 상태이므로 여부 자체를 보존
    pgs_val = df_new['착상 전 유전 검사 사용 여부'].fillna(0)
    pgs_cycle = df_new['PGS 시술 여부'].fillna(0)
    df_new['is_pgs_tested'] = ((pgs_val == 1.0) | (pgs_cycle == 1.0)).astype(int)

    pgd_val = df_new['착상 전 유전 진단 사용 여부'].fillna(0)
    pgd_cycle = df_new['PGD 시술 여부'].fillna(0)
    df_new['is_pgd_tested'] = ((pgd_val == 1.0) | (pgd_cycle == 1.0)).astype(int)

    # 기존 원본 컬럼 중 수치화 및 임베딩 완료된 문자열 컬럼 제거
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

print(f"Total features: {len(features)}")
print(f"Categorical features: {len(cat_cols)}")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================== 5. Model 1: LGBM A ====================
print("\n=== Training Model 1: LightGBM A ===")
lgb_a_oof = np.zeros(len(X))
lgb_a_test = np.zeros(len(X_test))

lgb_params_a = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 9,
    'scale_pos_weight': 1.5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va)
    
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    
    model = lgb.train(
        lgb_params_a,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks
    )
    
    lgb_a_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    lgb_a_test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

lgb_a_score = roc_auc_score(y, lgb_a_oof)
print(f"LGB_A OOF ROC-AUC: {lgb_a_score:.5f}")


# ==================== 6. Model 2: LGBM B ====================
print("\n=== Training Model 2: LightGBM B ===")
lgb_b_oof = np.zeros(len(X))
lgb_b_test = np.zeros(len(X_test))

lgb_params_b = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'scale_pos_weight': 1.0,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'random_state': 2026,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va)
    
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    
    model = lgb.train(
        lgb_params_b,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks
    )
    
    lgb_b_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    lgb_b_test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

lgb_b_score = roc_auc_score(y, lgb_b_oof)
print(f"LGB_B OOF ROC-AUC: {lgb_b_score:.5f}")


# ==================== 7. Model 3: XGBoost ====================
print("\n=== Training Model 3: XGBoost ===")
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


# ==================== 8. Model 4: CatBoost ====================
print("\n=== Training Model 4: CatBoost ===")
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
        verbose=False,
        thread_count=-1
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


# ==================== 9. Model 5: AdaBoost ====================
print("\n=== Training Model 5: AdaBoost ===")
ada_oof = np.zeros(len(X))
ada_test = np.zeros(len(X_test))

# AdaBoost용 범주형 피처 라벨 인코딩 수치화
X_ada = X.copy()
X_test_ada = X_test.copy()
for col in cat_cols:
    X_ada[col] = X_ada[col].cat.codes
    X_test_ada[col] = X_test_ada[col].cat.codes

for fold, (train_idx, val_idx) in enumerate(folds.split(X_ada, y)):
    X_tr, y_tr = X_ada.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_ada.iloc[val_idx], y.iloc[val_idx]
    
    # 수치 결측치 보간
    X_tr_filled = X_tr.fillna(-1)
    X_va_filled = X_va.fillna(-1)
    X_test_filled = X_test_ada.fillna(-1)
    
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    
    model.fit(X_tr_filled, y_tr)
    
    ada_oof[val_idx] = model.predict_proba(X_va_filled)[:, 1]
    ada_test += model.predict_proba(X_test_filled)[:, 1] / folds.n_splits

ada_score = roc_auc_score(y, ada_oof)
print(f"AdaBoost OOF ROC-AUC: {ada_score:.5f}")


# ==================== 10. Stacking (L2-Regularized Logistic Regression Meta-Learner) ====================
print("\n=== Stacking with Logistic Regression Meta-Learner ===")
# 메타 데이터셋 준비 (5개 베이스 모델 예측 결과 소프트 결합)
OOF_meta = np.column_stack([lgb_a_oof, lgb_b_oof, xgb_oof, cat_oof, ada_oof])
Test_meta = np.column_stack([lgb_a_test, lgb_b_test, xgb_test, cat_test, ada_test])

meta_oof = np.zeros(len(X))
meta_test = np.zeros(len(X_test))

# Meta-learner: 과적합 방지를 위해 강력한 L2 패널티를 주입한 Logistic Regression 사용
# 베이스 모델 예측값 간의 다중공선성을 억제하기 위해 C값을 조절 (C=0.1)
meta_clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)

for fold, (train_idx, val_idx) in enumerate(folds.split(OOF_meta, y)):
    X_tr, y_tr = OOF_meta[train_idx], y.iloc[train_idx]
    X_va, y_va = OOF_meta[val_idx], y.iloc[val_idx]
    
    meta_clf.fit(X_tr, y_tr)
    
    meta_oof[val_idx] = meta_clf.predict_proba(X_va)[:, 1]
    meta_test += meta_clf.predict_proba(Test_meta)[:, 1] / folds.n_splits

stacking_score = roc_auc_score(y, meta_oof)
print(f"Stacking Ensemble OOF ROC-AUC Score: {stacking_score:.5f}")

# 가중 평균 (Soft Voting Reference) 성능 측정용
w_lgb_a, w_lgb_b, w_xgb, w_cat, w_ada = 0.40, 0.00, 0.06, 0.30, 0.24
voting_oof = (w_lgb_a * lgb_a_oof) + (w_lgb_b * lgb_b_oof) + (w_xgb * xgb_oof) + (w_cat * cat_oof) + (w_ada * ada_oof)
voting_score = roc_auc_score(y, voting_oof)
print(f"Soft Voting Ensemble Reference OOF ROC-AUC Score: {voting_score:.5f}")

# 11. 최종 제출 파일 저장 (Stacking 예측 확률값 그대로 제출)
submission['probability'] = meta_test

output_sub_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v8_advanced.csv")
submission.to_csv(output_sub_path, index=False)
print(f"Saved Stacking Ensemble submission to: {output_sub_path}")
