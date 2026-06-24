import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUB_PATH)

target_col = '임신 성공 여부'

# 2. 진짜 결측 대치용 Train 통계치 산정
train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0  # 폴백

print(f"Calculated Train Actual Transfer Median Days: {train_transfer_median}")

# 3. 전처리 및 피처 엔지니어링 함수 (v16 임상 구조화 및 이원 결측 처리)
def preprocess_data(df, transfer_median):
    df_new = df.copy()
    
    # 3.1 특정 시술 유형 보간 (확실한 도메인 역산)
    df_new.loc[(df_new['특정 시술 유형'] == 'Unknown') & (df_new['미세주입된 난자 수'] > 0), '특정 시술 유형'] = 'ICSI'
    
    # 3.2 임상 프로세스 상태(Clinical State) 정의 및 추출
    ai_mask = (df_new['시술 유형'] == 'DI') | df_new['특정 시술 유형'].astype(str).str.contains('IUI|ICI|IVI|Generic DI', regex=True)
    frozen_mask = (df_new['동결 배아 사용 여부'] == 1.0) | (df_new['특정 시술 유형'] == 'Unknown')
    fresh_mask = (~ai_mask) & (~frozen_mask)
    
    df_new['clinical_state'] = 'Other'
    df_new.loc[ai_mask, 'clinical_state'] = 'IUI_cycle'
    
    # Fresh IVF 세분화
    fresh_cancelled_retrieval = fresh_mask & df_new['난자 채취 경과일'].isna()
    df_new.loc[fresh_cancelled_retrieval, 'clinical_state'] = 'Fresh_IVF_cancelled_at_retrieval'
    
    fresh_cancelled_mix = fresh_mask & df_new['난자 채취 경과일'].notna() & df_new['난자 혼합 경과일'].isna()
    df_new.loc[fresh_cancelled_mix, 'clinical_state'] = 'Fresh_IVF_cancelled_at_mix'
    
    transfer_missing = df_new['배아 이식 경과일'].isna() | df_new['이식된 배아 수'].isna() | (df_new['이식된 배아 수'] == 0)
    fresh_cancelled_transfer = fresh_mask & df_new['난자 혼합 경과일'].notna() & transfer_missing
    df_new.loc[fresh_cancelled_transfer, 'clinical_state'] = 'Fresh_IVF_cancelled_before_transfer'
    
    fresh_completed = fresh_mask & df_new['배아 이식 경과일'].notna() & df_new['이식된 배아 수'].notna() & (df_new['이식된 배아 수'] > 0)
    df_new.loc[fresh_completed, 'clinical_state'] = 'Fresh_IVF_completed'
    
    # Frozen IVF 세분화
    frozen_cancelled_transfer = frozen_mask & transfer_missing
    df_new.loc[frozen_cancelled_transfer, 'clinical_state'] = 'Frozen_IVF_cancelled_before_transfer'
    
    frozen_completed = frozen_mask & df_new['배아 이식 경과일'].notna() & df_new['이식된 배아 수'].notna() & (df_new['이식된 배아 수'] > 0)
    df_new.loc[frozen_completed, 'clinical_state'] = 'Frozen_IVF_completed'

    # 3.3 이원 결측 플래그(Dual Missingness Indicator) 생성
    # (1) 신선 난자 관련 컬럼
    fresh_egg_cols = ["수집된 신선 난자 수", "난자 채취 경과일", "혼합된 난자 수", "난자 혼합 경과일"]
    for col in fresh_egg_cols:
        df_new[f'{col}_not_applicable'] = (ai_mask | frozen_mask).astype(float)
        df_new[f'{col}_failed_or_missing'] = (fresh_mask & df_new[col].isna()).astype(float)
        
    # (2) 이식 경과일 / 이식 배아 수 컬럼
    transfer_cols = ["배아 이식 경과일", "이식된 배아 수"]
    for col in transfer_cols:
        df_new[f'{col}_not_applicable'] = ai_mask.astype(float)
        is_cancelled = df_new['clinical_state'].isin([
            'Fresh_IVF_cancelled_before_transfer', 'Frozen_IVF_cancelled_before_transfer', 
            'Fresh_IVF_cancelled_at_retrieval', 'Fresh_IVF_cancelled_at_mix'
        ])
        df_new[f'{col}_failed_or_missing'] = (is_cancelled & df_new[col].isna()).astype(float)

    # (3) 해동 배아 관련 컬럼
    frozen_embryo_cols = ["해동된 배아 수", "배아 해동 경과일"]
    for col in frozen_embryo_cols:
        df_new[f'{col}_not_applicable'] = (ai_mask | fresh_mask).astype(float)
        df_new[f'{col}_failed_or_missing'] = (frozen_mask & df_new[col].isna()).astype(float)

    # 3.4 배아 이식 경과일 수동 대치 (진짜 결측만 중앙값 대치)
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median
    df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)
    
    # 3.5 난자 및 정자 기증자 나이 결측치 최빈값 대치
    df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
    df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'
    
    # 3.6 나이 Ordinal 매핑
    age_map = {
        '만18-34세': 0, '만35-37세': 1, '만38-39세': 2, 
        '만40-42세': 3, '만43-44세': 4, '만45-50세': 5, '알 수 없음': np.nan
    }
    donor_age_map = {
        '만20세 이하': 0, '만21-25세': 1, '만26-30세': 2, 
        '만31-35세': 3, '만36-40세': 4, '만41-45세': 5, '알 수 없음': np.nan
    }
    df_new['시술 당시 나이_ordinal'] = df_new['시술 당시 나이'].map(age_map)
    df_new['난자 기증자 나이_ordinal'] = df_new['난자 기증자 나이'].map(donor_age_map)
    df_new['정자 기증자 나이_ordinal'] = df_new['정자 기증자 나이'].map(donor_age_map)
    
    # 3.7 고령 임산부 기준 플래그
    df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(float)
    df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(float)
    
    # 3.8 시술/임신/출산 횟수 문자열 -> 정수형 변환
    count_map = {'0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6}
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        df_new[f'{col}_int'] = df_new[col].map(count_map)
        
    # 3.9 파생 피처: 고령 임산부 및 기증 난자 상호작용 피처
    df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(float)
    df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(float)
    df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(float)
    
    # 3.10 파생 피처: 배아 배양 일수 계산 (배양 일수는 양수이고 유효할 때만 계산, 나머지는 -1 격리)
    df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']
    df_new.loc[df_new['배아 이식 경과일'] <= 0, 'embryo_culture_days'] = -1.0
    df_new.loc[df_new['난자 혼합 경과일'].isna(), 'embryo_culture_days'] = -1.0
    
    # 3.11 남성/여성 불임 요인 플래그
    male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
    df_new['is_male_infertility'] = df_new[male_factors].any(axis=1).astype(int)
    
    female_factors = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '여성 주 불임 원인', '여성 부 불임 원인']
    df_new['is_female_infertility'] = df_new[female_factors].any(axis=1).astype(int)
    
    # 3.12 실질 가임 연령 및 리쥬브네이션 갭 피처 추가
    patient_mid = {
        '만18-34세': 31, '만35-37세': 36, '만38-39세': 38.5, '만40-42세': 41,
        '만43-44세': 43.5, '만45-50세': 47.5, '알 수 없음': np.nan
    }
    donor_mid = {
        '만20세 이하': 20, '만21-25세': 23, '만26-30세': 28, '만31-35세': 33,
        '만36-40세': 38, '만41-45세': 43, '만46세 이상': 47, '알 수 없음': np.nan
    }
    df_new['patient_age_mid'] = df_new['시술 당시 나이'].map(patient_mid)
    df_new['oocyte_donor_age_mid'] = df_new['난자 기증자 나이'].map(donor_mid)
    
    donor_known = (df_new['난자 출처'] == '기증 제공') & df_new['oocyte_donor_age_mid'].notna()
    df_new['effective_maternal_age'] = df_new['patient_age_mid']
    df_new.loc[donor_known, 'effective_maternal_age'] = df_new.loc[donor_known, 'oocyte_donor_age_mid']
    
    df_new['donor_rejuvenation_gap'] = 0.0
    df_new.loc[donor_known, 'donor_rejuvenation_gap'] = (
        df_new.loc[donor_known, 'patient_age_mid'] - df_new.loc[donor_known, 'oocyte_donor_age_mid']
    )
    df_new['donor_rejuvenation_gap_positive'] = (df_new['donor_rejuvenation_gap'] > 0).astype(int)
    df_new['donor_rejuvenation_gap_10plus'] = (df_new['donor_rejuvenation_gap'] >= 10).astype(int)
    
    # 3.13 조건부 파생 변수 (수정율 및 수정란 형성 효율)
    df_new['fertilization_rate'] = -1.0
    valid_retrieval = (df_new['수집된 신선 난자 수'] > 0)
    df_new.loc[valid_retrieval, 'fertilization_rate'] = df_new.loc[valid_retrieval, '혼합된 난자 수'] / df_new.loc[valid_retrieval, '수집된 신선 난자 수']
    
    df_new['embryo_formation_rate'] = -1.0
    valid_mix = (df_new['혼합된 난자 수'] > 0)
    df_new.loc[valid_mix, 'embryo_formation_rate'] = df_new.loc[valid_mix, '총 생성 배아 수'] / df_new.loc[valid_mix, '혼합된 난자 수']

    # 원래 나이 컬럼 및 횟수 컬럼 드롭
    cols_to_drop = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + count_cols
    df_new = df_new.drop(columns=cols_to_drop)
    
    return df_new

# 전처리 수행
print("Preprocessing datasets...")
X = preprocess_data(train, train_transfer_median)
X_test = preprocess_data(test, train_transfer_median)

drop_cols = ['ID', target_col] if target_col in X.columns else ['ID']
features = [col for col in X.columns if col not in drop_cols]
y = train[target_col]

X = X[features].copy()
X_test = X_test[features].copy()

# 범주형 컬럼 확인 및 Train + Test 통합 카테고리 일치
cat_cols = []
for col in features:
    if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
        cat_cols.append(col)
        X[col] = X[col].astype(str).replace('nan', np.nan)
        X_test[col] = X_test[col].astype(str).replace('nan', np.nan)
        
        unique_cats = sorted(list(set(X[col].dropna().unique().tolist() + X_test[col].dropna().unique().tolist())))
        X[col] = pd.Categorical(X[col], categories=unique_cats)
        X_test[col] = pd.Categorical(X_test[col], categories=unique_cats)

print(f"Total features: {len(features)}")
print(f"Categorical features: {len(cat_cols)}")

# ==================== ECDF 랭크 블렌딩 클래스 구현 ====================
class ECDFReference:
    def __init__(self, ref):
        self.sorted = np.sort(np.asarray(ref, dtype=float))
        self.n = max(len(self.sorted), 1)

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        left = np.searchsorted(self.sorted, x, side="left")
        right = np.searchsorted(self.sorted, x, side="right")
        ranks = (left + right) / 2.0
        return ranks / self.n

# ==================== 범주형 정수 변환 유틸리티 (CatBoost & MLP 속도 단축용) ====================
def ordinal_encode_cats(X_tr, X_va, test_df, cat_columns):
    X_tr_f, X_va_f, test_f = X_tr.copy(), X_va.copy(), test_df.copy()
    if cat_columns:
        for c in cat_columns:
            X_tr_f[c] = X_tr_f[c].astype(str).where(X_tr_f[c].notna(), "미기록")
            X_va_f[c] = X_va_f[c].astype(str).where(X_va_f[c].notna(), "미기록")
            test_f[c] = test_f[c].astype(str).where(test_f[c].notna(), "미기록")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        X_tr_f[cat_columns] = enc.fit_transform(X_tr_f[cat_columns])
        X_va_f[cat_columns] = enc.transform(X_va_f[cat_columns])
        test_f[cat_columns] = enc.transform(test_f[cat_columns])
    return X_tr_f, X_va_f, test_f

# ==================== 모델 예측 캐싱 설정 ====================
os.makedirs(os.path.join(BASE_DIR, "src", "v16", "checkpoints"), exist_ok=True)
oof_cache_path = os.path.join(BASE_DIR, "src", "v16", "checkpoints", "model_oofs_seeds.pkl")

seeds = [42, 1004, 7, 2026, 88]
model_names = ['lgbm', 'xgboost', 'catboost', 'mlp']

if os.path.exists(oof_cache_path):
    print(f"\n[Cache] Loading multi-seed model predictions from {oof_cache_path}...")
    with open(oof_cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    model_oofs_seeds = cache_data['model_oofs_seeds']
    model_tests_seeds = cache_data['model_tests_seeds']
else:
    model_oofs_seeds = {seed: {} for seed in seeds}
    model_tests_seeds = {seed: {} for seed in seeds}

    for seed in seeds:
        print(f"\n=======================================================")
        print(f"               STARTING SEED: {seed}                   ")
        print(f"=======================================================")
        
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # 1. LightGBM (category dtypes)
        print(f"\n--- [Seed {seed}] Training Model 1: LightGBM ---")
        lgb_oof = np.zeros(len(X))
        lgb_test = np.zeros(len(X_test))

        X_lgb = X.copy()
        X_test_lgb = X_test.copy()
        for col in cat_cols:
            X_lgb[col] = X_lgb[col].astype('category')
            X_test_lgb[col] = X_test_lgb[col].astype('category')

        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 44,
            'max_depth': 6,
            'min_child_samples': 82,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1.1,
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1
        }

        for fold, (train_idx, val_idx) in enumerate(folds.split(X_lgb, y)):
            X_tr, y_tr = X_lgb.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_lgb.iloc[val_idx], y.iloc[val_idx]
            
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
            
            model = lgb.train(
                lgb_params, trn_data, num_boost_round=2000,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            lgb_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
            lgb_test += model.predict(X_test_lgb, num_iteration=model.best_iteration) / folds.n_splits

        print(f"[Seed {seed}] LightGBM OOF AUC: {roc_auc_score(y, lgb_oof):.6f}")
        model_oofs_seeds[seed]['lgbm'] = lgb_oof
        model_tests_seeds[seed]['lgbm'] = lgb_test

        # 2. XGBoost (enable_categorical)
        print(f"\n--- [Seed {seed}] Training Model 2: XGBoost ---")
        xgb_oof = np.zeros(len(X))
        xgb_test = np.zeros(len(X_test))

        xgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.025,
            'max_depth': 4,
            'min_child_weight': 8,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1.1,
            'tree_method': 'hist',
            'enable_categorical': True,
            'random_state': seed,
            'n_jobs': -1,
            'eval_metric': 'auc',
            'early_stopping_rounds': 50
        }

        for fold, (train_idx, val_idx) in enumerate(folds.split(X_lgb, y)):
            X_tr, y_tr = X_lgb.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_lgb.iloc[val_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            xgb_oof[val_idx] = model.predict_proba(X_va)[:, 1]
            xgb_test += model.predict_proba(X_test_lgb)[:, 1] / folds.n_splits

        print(f"[Seed {seed}] XGBoost OOF AUC: {roc_auc_score(y, xgb_oof):.6f}")
        model_oofs_seeds[seed]['xgboost'] = xgb_oof
        model_tests_seeds[seed]['xgboost'] = xgb_test

        # 3. CatBoost (CPU 최적화를 위해 Ordinal Encoding 적용)
        print(f"\n--- [Seed {seed}] Training Model 3: CatBoost (Ordinal Encoded) ---")
        cat_oof = np.zeros(len(X))
        cat_test = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            X_tr_enc, X_va_enc, test_enc = ordinal_encode_cats(X_tr, X_va, X_test, cat_cols)
            
            model = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.03,
                depth=6,
                auto_class_weights='Balanced',
                eval_metric='AUC',
                random_seed=seed,
                task_type='CPU',
                early_stopping_rounds=50,
                verbose=False
            )
            
            model.fit(X_tr_enc, y_tr, eval_set=(X_va_enc, y_va), use_best_model=True)
            
            cat_oof[val_idx] = model.predict_proba(X_va_enc)[:, 1]
            cat_test += model.predict_proba(test_enc)[:, 1] / folds.n_splits

        print(f"[Seed {seed}] CatBoost OOF AUC: {roc_auc_score(y, cat_oof):.6f}")
        model_oofs_seeds[seed]['catboost'] = cat_oof
        model_tests_seeds[seed]['catboost'] = cat_test

        # 4. MLPClassifier
        print(f"\n--- [Seed {seed}] Training Model 4: MLPClassifier ---")
        mlp_oof = np.zeros(len(X))
        mlp_test = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            X_tr_enc, X_va_enc, test_enc = ordinal_encode_cats(X_tr, X_va, X_test, cat_cols)
            
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=4096,
                    learning_rate_init=0.001,
                    max_iter=80,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=8,
                    random_state=seed + fold,
                    verbose=False
                )
            )
            
            pipe.fit(X_tr_enc, y_tr)
            mlp_oof[val_idx] = pipe.predict_proba(X_va_enc)[:, 1]
            mlp_test += pipe.predict_proba(test_enc)[:, 1] / folds.n_splits

        print(f"[Seed {seed}] MLPClassifier OOF AUC: {roc_auc_score(y, mlp_oof):.6f}")
        model_oofs_seeds[seed]['mlp'] = mlp_oof
        model_tests_seeds[seed]['mlp'] = mlp_test

    # 캐시 저장
    cache_data = {
        'model_oofs_seeds': model_oofs_seeds,
        'model_tests_seeds': model_tests_seeds
    }
    with open(oof_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"\n[Cache] Saved multi-seed predictions to {oof_cache_path}")

# ==================== 5. Leak-Free ECDF Rank Bagging & Weight Optimization ====================
print("\n--- Applying Multi-Seed Leak-Free ECDF Rank Bagging & Optimization ---")

# 랭크 변환 컨테이너
mean_rank_oofs = {name: np.zeros(len(y)) for name in model_names}
mean_rank_tests = {name: np.zeros(len(test)) for name in model_names}

for seed in seeds:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # 이 시드에서의 랭크 변환 저장
    seed_rank_oofs = {name: np.zeros(len(y)) for name in model_names}
    seed_rank_tests = {name: np.zeros(len(test)) for name in model_names}
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        for name in model_names:
            # 훈련 fold 예측 분포로 ECDFReference 학습
            oof_train_preds = model_oofs_seeds[seed][name][train_idx]
            ecdf = ECDFReference(oof_train_preds)
            
            # Validation OOF 변환
            oof_val_preds = model_oofs_seeds[seed][name][val_idx]
            seed_rank_oofs[name][val_idx] = ecdf.transform(oof_val_preds)
            
            # Test 변환
            test_fold_preds = model_tests_seeds[seed][name]
            seed_rank_tests[name] += ecdf.transform(test_fold_preds) / folds.n_splits
            
    # 시드간 랭크값 평균 누적 (배깅)
    for name in model_names:
        mean_rank_oofs[name] += seed_rank_oofs[name] / len(seeds)
        mean_rank_tests[name] += seed_rank_tests[name] / len(seeds)

print("\nMulti-seed Bagged Rank OOF AUCs:")
for name in model_names:
    score = roc_auc_score(y, mean_rank_oofs[name])
    print(f"  - {name} Bagged Rank OOF AUC: {score:.6f}")

# --- Scipy 가중치 최적화 (Nelder-Mead + Softmax 방식) ---
print("\nOptimizing ensemble weights on Bagged ranks using Scipy Nelder-Mead...")

def objective(x):
    # Softmax 매핑
    exp_x = np.exp(x - np.max(x))
    w = exp_x / np.sum(exp_x)
    
    blended_oof = np.zeros(len(y))
    for idx, name in enumerate(model_names):
        blended_oof += w[idx] * mean_rank_oofs[name]
        
    return -roc_auc_score(y, blended_oof)

init_x = [0.0] * len(model_names)

res = minimize(
    objective,
    init_x,
    method='Nelder-Mead',
    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6}
)

optimized_x = res.x
exp_opt = np.exp(optimized_x - np.max(optimized_x))
optimized_weights = exp_opt / np.sum(exp_opt)

print("\nOptimization Complete!")
for idx, name in enumerate(model_names):
    print(f"  - {name} Optimized Weight: {optimized_weights[idx]:.4f}")

# 최종 가중 평균 블렌드
final_oof_rank = np.zeros(len(y))
final_test_rank = np.zeros(len(test))

for idx, name in enumerate(model_names):
    final_oof_rank += optimized_weights[idx] * mean_rank_oofs[name]
    final_test_rank += optimized_weights[idx] * mean_rank_tests[name]

final_score = roc_auc_score(y, final_oof_rank)
print(f"\nFinal Optimized ECDF Bagged Rank OOF AUC: {final_score:.6f}")

# ==================== 6. Save Submission ====================
submission['probability'] = final_test_rank
output_sub_dir = os.path.join(BASE_DIR, "src", "v16")
os.makedirs(output_sub_dir, exist_ok=True)

output_sub_path = os.path.join(output_sub_dir, f"submission_v16_bag_{final_score:.6f}.csv")
submission.to_csv(output_sub_path, index=False)
print(f"\nSaved v16 ECDF Bagged Submission to: {output_sub_path}")

# 백업용으로 데이터 폴더에도 저장
backup_sub_path = os.path.join(DATA_DIR, f"submission_v16_bag_{final_score:.6f}.csv")
submission.to_csv(backup_sub_path, index=False)
print(f"Saved backup submission to: {backup_sub_path}")
