import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge, LinearRegression

import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

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

# 2. 진짜 결측 대치용 Train 통계치 산정 (v16)
train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0
print(f"Calculated Train Actual Transfer Median Days: {train_transfer_median}")

# 3. 전처리 및 피처 엔지니어링 함수
def build_features(df, transfer_median):
    df_new = df.copy()
    
    # 3.1 특정 시술 유형 보간 (v16)
    if '특정 시술 유형' in df_new.columns and '미세주입된 난자 수' in df_new.columns:
        df_new.loc[(df_new['특정 시술 유형'] == 'Unknown') & (df_new['미세주입된 난자 수'] > 0), '특정 시술 유형'] = 'ICSI'
        
    # 3.2 임상 프로세스 상태(Clinical State) 정의 및 추출 (v16)
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

    # 3.3 Teammate's missingness indicators for embryo count/binary cols:
    df_new["is_DI"] = (df_new["시술 유형"] == "DI").astype(int)
    df_new["froze_embryo"] = df_new["동결 배아 사용 여부"].fillna(0).astype(int)
    
    EMBRYO_COUNT_COLS = [
        "총 생성 배아 수","미세주입된 난자 수","미세주입에서 생성된 배아 수","이식된 배아 수",
        "미세주입 배아 이식 수","저장된 배아 수","미세주입 후 저장된 배아 수","해동된 배아 수",
        "해동 난자 수","수집된 신선 난자 수","저장된 신선 난자 수","혼합된 난자 수",
        "파트너 정자와 혼합된 난자 수","기증자 정자와 혼합된 난자 수"
    ]
    EMBRYO_BINARY_COLS = [
        "단일 배아 이식 여부","착상 전 유전 진단 사용 여부","동결 배아 사용 여부",
        "신선 배아 사용 여부","기증 배아 사용 여부","대리모 여부"
    ]
    
    embryo_cols_present = [c for c in (EMBRYO_COUNT_COLS + EMBRYO_BINARY_COLS) if c in df_new.columns]
    for col in embryo_cols_present:
        df_new[f"{col}_missing"] = df_new[col].isna().astype(int)
        is_di_missing = (df_new["시술 유형"] == "DI") & df_new[col].isna()
        df_new[f"{col}_not_applicable_DI"] = is_di_missing.astype(int)

    # 3.4 v16's Dual missingness flags:
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

    # 3.5 배아 이식 경과일 수동 대치 (진짜 결측만 중앙값 대치)
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median

    # 3.6 Outlier Clipping (teammate)
    CLIP_RULES = {
        "총 생성 배아 수": 40, "수집된 신선 난자 수": 40, "미세주입된 난자 수": 45,
        "혼합된 난자 수": 40, "저장된 배아 수": 30, "배아 이식 경과일": 7, "난자 혼합 경과일": 7,
        "배아 해동 경과일": 2, "난자 해동 경과일": 1
    }
    for col, upper in CLIP_RULES.items():
        if col in df_new.columns:
            df_new[f"{col}_high_flag"] = (df_new[col] > upper).astype(int)
            df_new[col] = df_new[col].clip(upper=upper)

    # 3.7 배아 이식 경과일 플래그 (teammate)
    if "배아 이식 경과일" in df_new.columns:
        df_new["transfer_day_0_1_flag"] = df_new["배아 이식 경과일"].isin([0, 1]).astype(int)
        df_new["transfer_day_3_flag"] = (df_new["배아 이식 경과일"] == 3).astype(int)
        df_new["transfer_day_5_or_more_flag"] = (df_new["배아 이식 경과일"] >= 5).astype(int)

    # 3.8 6 Ratios using safe_ratio (teammate)
    RATIO_SPECS = [
        ("총 생성 배아 수", "혼합된 난자 수", "fertilization_rate"),
        ("미세주입에서 생성된 배아 수", "미세주입된 난자 수", "icsi_fertilization_rate"),
        ("이식된 배아 수", "총 생성 배아 수", "embryo_utilization_rate"),
        ("저장된 배아 수", "총 생성 배아 수", "embryo_freezing_rate"),
        ("혼합된 난자 수", "수집된 신선 난자 수", "oocyte_utilization_rate"),
        ("이식된 배아 수", "해동된 배아 수", "thawed_embryo_transfer_ratio"),
    ]
    for num, den, new in RATIO_SPECS:
        if num in df_new.columns and den in df_new.columns:
            can = df_new[num].notna() & df_new[den].notna() & (df_new[den] > 0)
            df_new[f"{new}_available"] = can.astype(int)
            df_new[new] = np.where(can, df_new[num] / df_new[den], np.nan)

    # 3.9 실효 모성 나이 및 리쥬브네이션 갭 (teammate)
    patient_mid = {"만18-34세": 31, "만35-37세": 36, "만38-39세": 38.5, "만40-42세": 41,
                    "만43-44세": 43.5, "만45-50세": 47.5, "알 수 없음": np.nan}
    donor_mid = {"만20세 이하": 20, "만21-25세": 23, "만26-30세": 28, "만31-35세": 33,
                 "만36-40세": 38, "만41-45세": 43, "알 수 없음": np.nan}
                 
    if "난자 출처" in df_new.columns and "시술 당시 나이" in df_new.columns:
        df_new["patient_age_mid"] = df_new["시술 당시 나이"].map(patient_mid)
        donor_age_mid = df_new["난자 기증자 나이"].map(donor_mid) if "난자 기증자 나이" in df_new.columns else pd.Series(np.nan, index=df_new.index)
        donor_known = (df_new["난자 출처"] == "기증 제공") & donor_age_mid.notna()
        df_new["effective_maternal_age_mid"] = df_new["patient_age_mid"]
        df_new.loc[donor_known, "effective_maternal_age_mid"] = donor_age_mid[donor_known]
        df_new["donor_rejuvenation_gap"] = 0.0
        df_new.loc[donor_known, "donor_rejuvenation_gap"] = (
            df_new.loc[donor_known, "patient_age_mid"] - donor_age_mid[donor_known]
        )

    # 3.10 나이 Ordinal 매핑 & 플래그 (v16)
    age_map = {
        '만18-34세': 0, '만35-37세': 1, '만38-39세': 2, 
        '만40-42세': 3, '만43-44세': 4, '만45-50세': 5, '알 수 없음': np.nan
    }
    donor_age_map = {
        '만20세 이하': 0, '만21-25세': 1, '만26-30세': 2, 
        '만31-35세': 3, '만36-40세': 4, '만41-45세': 5, '알 수 없음': np.nan
    }
    
    # 난자/정자 기증자 나이 결측치 최빈값 대치 (v16)
    df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
    df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'

    if '시술 당시 나이' in df_new.columns:
        df_new['시술 당시 나이_ordinal'] = df_new['시술 당시 나이'].map(age_map)
        df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(float)
        df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(float)
        df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(float)
        df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(float)
        
    if '난자 기증자 나이' in df_new.columns:
        df_new['난자 기증자 나이_ordinal'] = df_new['난자 기증자 나이'].map(donor_age_map)
        df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(float)
        
    if '정자 기증자 나이' in df_new.columns:
        df_new['정자 기증자 나이_ordinal'] = df_new['정자 기증자 나이'].map(donor_age_map)

    # 3.11 시술유형 세부 토큰화 (teammate)
    if "특정 시술 유형" in df_new.columns:
        s = df_new["특정 시술 유형"].astype(str)
        for token in ["IVF", "ICSI", "IUI", "ICI", "GIFT", "FER", "Generic DI", "IVI", "BLASTOCYST", "AH"]:
            safe = token.lower().replace(" ", "_")
            df_new[f"spec_has_{safe}"] = s.str.contains(token, regex=False, na=False).astype(int)

    # 3.12 나이 & 난자 출처 인터랙션 (teammate)
    if {"시술 당시 나이", "난자 출처"}.issubset(df_new.columns):
        df_new["age_oocyte_source"] = df_new["시술 당시 나이"].astype(str) + "_" + df_new["난자 출처"].astype(str)

    # 3.13 시술/임신/출산 횟수 문자열 -> 정수형 변환 (v16)
    count_map = {'0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6}
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        if col in df_new.columns:
            df_new[f'{col}_int'] = df_new[col].map(count_map)

    # 3.14 배아 배양 일수 계산 (v16)
    if "배아 이식 경과일" in df_new.columns and "난자 혼합 경과일" in df_new.columns:
        df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']

    # 3.15 남성/여성 불임 요인 플래그 (v16)
    male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
    male_present = [f for f in male_factors if f in df_new.columns]
    if male_present:
        df_new['is_male_infertility'] = df_new[male_present].any(axis=1).astype(int)
        
    female_factors = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '여성 주 불임 원인', '여성 부 불임 원인']
    female_present = [f for f in female_factors if f in df_new.columns]
    if female_present:
        df_new['is_female_infertility'] = df_new[female_present].any(axis=1).astype(int)

    # 원래 나이 및 횟수 컬럼 제거
    cols_to_drop = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + count_cols
    df_new = df_new.drop(columns=[c for c in cols_to_drop if c in df_new.columns], errors='ignore')
    
    # 3.16 Dead Columns 제거
    DEAD_COLS = ["불임 원인 - 여성 요인", "난자 채취 경과일"]
    df_new = df_new.drop(columns=[c for c in DEAD_COLS if c in df_new.columns], errors='ignore')

    return df_new

# 전처리 적용
print("Preprocessing datasets...")
X = build_features(train, train_transfer_median)
X_test = build_features(test, train_transfer_median)

drop_cols = ['ID', target_col] if target_col in X.columns else ['ID']
features = [col for col in X.columns if col not in drop_cols]
y = train[target_col].values.astype(int)
test_ids = test['ID'].values

X = X[features].copy()
X_test = X_test[features].copy()

# 범주형 컬럼 확인 및 Train 고정 범주 적용 (DACON 규정 완벽 준수)
SENTINEL = "정보없음"
cat_cols = []
for col in features:
    if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
        cat_cols.append(col)
        X[col] = X[col].astype(str).replace('nan', np.nan).fillna(SENTINEL)
        X_test[col] = X_test[col].astype(str).replace('nan', np.nan).fillna(SENTINEL)
        
        train_categories = sorted(list(set(X[col].unique()) | {SENTINEL}))
        X[col] = pd.Categorical(X[col], categories=train_categories)
        X_test[col] = pd.Categorical(X_test[col], categories=train_categories)
        X_test[col] = X_test[col].fillna(SENTINEL)

print(f"Total features: {len(features)}")
print(f"Categorical features: {len(cat_cols)}")

# ==================== ECDF 랭크변환 클래스 구현 ====================
class ECDFReference:
    def __init__(self, ref):
        self.sorted = np.sort(np.asarray(ref, dtype=float))
        self.n = max(len(self.sorted), 1)

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        left = np.searchsorted(self.sorted, x, side="left")
        right = np.searchsorted(self.sorted, x, side="right")
        return (left + right) / 2.0 / self.n

# ==================== 범주형 정수 변환 유틸리티 (MLP 용) ====================
def ordinal_encode_cats(X_tr, X_va, test_df, cat_columns):
    X_tr_f, X_va_f, test_f = X_tr.copy(), X_va.copy(), test_df.copy()
    if cat_columns:
        for c in cat_columns:
            X_tr_f[c] = X_tr_f[c].astype(str)
            X_va_f[c] = X_va_f[c].astype(str)
            test_f[c] = test_f[c].astype(str)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        X_tr_f[cat_columns] = enc.fit_transform(X_tr_f[cat_columns])
        X_va_f[cat_columns] = enc.transform(X_va_f[cat_columns])
        test_f[cat_columns] = enc.transform(test_f[cat_columns])
    return X_tr_f, X_va_f, test_f

# ==================== 모델 학습 캐싱 설정 ====================
os.makedirs(os.path.join(BASE_DIR, "src", "v17", "checkpoints"), exist_ok=True)
oof_cache_path = os.path.join(BASE_DIR, "src", "v17", "checkpoints", "model_oofs_seeds.pkl")

seeds = [42, 1004, 7, 2026, 88]
model_names = ['lgbm', 'xgboost', 'catboost', 'catboost_weighted', 'lgbm_spw2', 'mlp']

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

if True:
    if os.path.exists(oof_cache_path):
        print(f"\n[Cache] Loading multi-seed model predictions from {oof_cache_path}...")
        with open(oof_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        model_oofs_seeds = cache_data['model_oofs_seeds']
        model_tests_seeds = cache_data['model_tests_seeds']
    else:
        model_oofs_seeds = {seed: {name: np.zeros(len(X)) for name in model_names} for seed in seeds}
        model_tests_seeds = {seed: {name: np.zeros(len(X_test)) for name in model_names} for seed in seeds}

    for seed in seeds:
        if np.any(model_oofs_seeds[seed]['lgbm'] != 0):
            print(f"\n[Cache] Seed {seed} already trained. Skipping...")
            continue
        print(f"\n=======================================================")
        print(f"               STARTING SEED: {seed}                   ")
        print(f"=======================================================")
        
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # Tree 모델용 카테고리 명시화
        X_tree = X.copy()
        X_test_tree = X_test.copy()
        for col in cat_cols:
            X_tree[col] = X_tree[col].astype('category')
            X_test_tree[col] = X_test_tree[col].astype('category')

        # 1. LightGBM
        print(f"\n--- [Seed {seed}] Model 1: LightGBM ---")
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
        for fold, (train_idx, val_idx) in enumerate(folds.split(X_tree, y)):
            X_tr, y_tr = X_tree.iloc[train_idx], y[train_idx]
            X_va, y_va = X_tree.iloc[val_idx], y[val_idx]
            
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
            
            model = lgb.train(
                lgb_params, trn_data, num_boost_round=2000,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            model_oofs_seeds[seed]['lgbm'][val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
            model_tests_seeds[seed]['lgbm'] += model.predict(X_test_tree, num_iteration=model.best_iteration) / folds.n_splits
        print(f"[Seed {seed}] LightGBM OOF AUC: {roc_auc_score(y, model_oofs_seeds[seed]['lgbm']):.6f}")

        # 2. XGBoost
        print(f"\n--- [Seed {seed}] Model 2: XGBoost ---")
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
        for fold, (train_idx, val_idx) in enumerate(folds.split(X_tree, y)):
            X_tr, y_tr = X_tree.iloc[train_idx], y[train_idx]
            X_va, y_va = X_tree.iloc[val_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            model_oofs_seeds[seed]['xgboost'][val_idx] = model.predict_proba(X_va)[:, 1]
            model_tests_seeds[seed]['xgboost'] += model.predict_proba(X_test_tree)[:, 1] / folds.n_splits
        print(f"[Seed {seed}] XGBoost OOF AUC: {roc_auc_score(y, model_oofs_seeds[seed]['xgboost']):.6f}")

        # 3. CatBoost (Standard)
        print(f"\n--- [Seed {seed}] Model 3: CatBoost (Standard) ---")
        cat_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': 1500,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 5,
            'random_seed': seed,
            'early_stopping_rounds': 100,
            'allow_writing_files': False,
            'verbose': False,
            'task_type': 'CPU'
        }
        for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
            tp = Pool(X_tr, y_tr, cat_features=cat_idx)
            vp = Pool(X_va, y_va, cat_features=cat_idx)
            test_pool = Pool(X_test, cat_features=cat_idx)
            
            model = CatBoostClassifier(**cat_params)
            model.fit(tp, eval_set=vp, use_best_model=True)
            
            model_oofs_seeds[seed]['catboost'][val_idx] = model.predict_proba(vp)[:, 1]
            model_tests_seeds[seed]['catboost'] += model.predict_proba(test_pool)[:, 1] / folds.n_splits
        print(f"[Seed {seed}] CatBoost OOF AUC: {roc_auc_score(y, model_oofs_seeds[seed]['catboost']):.6f}")

        # 4. CatBoost (Weighted)
        print(f"\n--- [Seed {seed}] Model 4: CatBoost (Weighted) ---")
        cat_weighted_params = cat_params.copy()
        cat_weighted_params['auto_class_weights'] = 'Balanced'
        for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
            tp = Pool(X_tr, y_tr, cat_features=cat_idx)
            vp = Pool(X_va, y_va, cat_features=cat_idx)
            test_pool = Pool(X_test, cat_features=cat_idx)
            
            model = CatBoostClassifier(**cat_weighted_params)
            model.fit(tp, eval_set=vp, use_best_model=True)
            
            model_oofs_seeds[seed]['catboost_weighted'][val_idx] = model.predict_proba(vp)[:, 1]
            model_tests_seeds[seed]['catboost_weighted'] += model.predict_proba(test_pool)[:, 1] / folds.n_splits
        print(f"[Seed {seed}] CatBoost Weighted OOF AUC: {roc_auc_score(y, model_oofs_seeds[seed]['catboost_weighted']):.6f}")

        # 5. LightGBM SPW2
        print(f"\n--- [Seed {seed}] Model 5: LightGBM SPW2 ---")
        lgb_spw2_params = {
            'objective': 'binary',
            'n_estimators': 3000,
            'learning_rate': 0.018,
            'num_leaves': 63,
            'min_child_samples': 60,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 2.0,
            'scale_pos_weight': 2.0,
            'random_state': seed,
            'verbose': -1,
            'n_jobs': -1
        }
        for fold, (train_idx, val_idx) in enumerate(folds.split(X_tree, y)):
            X_tr, y_tr = X_tree.iloc[train_idx], y[train_idx]
            X_va, y_va = X_tree.iloc[val_idx], y[val_idx]
            
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
            
            model = lgb.train(
                lgb_spw2_params, trn_data, num_boost_round=3000,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            model_oofs_seeds[seed]['lgbm_spw2'][val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
            model_tests_seeds[seed]['lgbm_spw2'] += model.predict(X_test_tree, num_iteration=model.best_iteration) / folds.n_splits
        print(f"[Seed {seed}] LightGBM SPW2 OOF AUC: {roc_auc_score(y, model_oofs_seeds[seed]['lgbm_spw2']):.6f}")

        # 6. MLPClassifier
        print(f"\n--- [Seed {seed}] Model 6: MLPClassifier ---")
        for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
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
            model_oofs_seeds[seed]['mlp'][val_idx] = pipe.predict_proba(X_va_enc)[:, 1]
            model_tests_seeds[seed]['mlp'] += pipe.predict_proba(test_enc)[:, 1] / folds.n_splits
        print(f"[Seed {seed}] MLPClassifier OOF AUC: {roc_auc_score(y, model_oofs_seeds[seed]['mlp']):.6f}")

        # 각 시드 완료 후 즉시 캐시 저장
        cache_data = {
            'model_oofs_seeds': model_oofs_seeds,
            'model_tests_seeds': model_tests_seeds
        }
        with open(oof_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"\n[Cache] Saved checkpoint after Seed {seed} to {oof_cache_path}")

# ==================== ECDF Rank Bagging ====================
print("\n--- Applying Multi-Seed Leak-Free ECDF Rank Bagging ---")

mean_rank_oofs = {name: np.zeros(len(y)) for name in model_names}
mean_rank_tests = {name: np.zeros(len(X_test)) for name in model_names}

for seed in seeds:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    seed_rank_oofs = {name: np.zeros(len(y)) for name in model_names}
    seed_rank_tests = {name: np.zeros(len(X_test)) for name in model_names}
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        for name in model_names:
            ref = ECDFReference(model_oofs_seeds[seed][name][train_idx])
            
            seed_rank_oofs[name][val_idx] = ref.transform(model_oofs_seeds[seed][name][val_idx])
            seed_rank_tests[name] += ref.transform(model_tests_seeds[seed][name]) / folds.n_splits
            
    for name in model_names:
        mean_rank_oofs[name] += seed_rank_oofs[name] / len(seeds)
        mean_rank_tests[name] += seed_rank_tests[name] / len(seeds)

print("\nMulti-seed Bagged Rank OOF AUCs:")
for name in model_names:
    score = roc_auc_score(y, mean_rank_oofs[name])
    print(f"  - {name} Bagged Rank OOF AUC: {score:.6f}")

# ==================== ECDF Ridge Stacking ====================
print("\n--- ECDF Ridge Stacking Optimization ---")

def make_ecdf_ridge_stack(names, mean_rank_oofs, mean_rank_tests, y, alpha=1.0, n_splits=5, seed=42):
    oof_matrix = np.column_stack([mean_rank_oofs[n] for n in names])
    test_matrix = np.column_stack([mean_rank_tests[n] for n in names])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y))
    test_pred = np.zeros(test_matrix.shape[0])
    ncol = len(names)

    for tr, va in skf.split(oof_matrix, y):
        Xtr = np.empty((len(tr), ncol))
        Xva = np.empty((len(va), ncol))
        Xte = np.empty((test_matrix.shape[0], ncol))
        
        for c in range(ncol):
            ref = ECDFReference(oof_matrix[tr, c])
            Xtr[:, c] = ref.transform(oof_matrix[tr, c])
            Xva[:, c] = ref.transform(oof_matrix[va, c])
            Xte[:, c] = ref.transform(test_matrix[:, c])

        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha) if alpha > 0 else LinearRegression())
        model.fit(Xtr, y[tr])
        
        oof_pred[va] = np.clip(model.predict(Xva), 0, 1)
        test_pred += np.clip(model.predict(Xte), 0, 1) / n_splits

    auc = roc_auc_score(y, oof_pred)
    return oof_pred, test_pred, auc

# alpha 그리드 비교
alpha_list = [0.0, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]
ridge_results = {}

print(f"{'alpha':>8} | {'OOF AUC':>8}")
print("-" * 21)
for alpha in alpha_list:
    _, _, auc = make_ecdf_ridge_stack(model_names, mean_rank_oofs, mean_rank_tests, y, alpha=alpha)
    ridge_results[alpha] = auc
    label = "0.0 (LR)" if alpha == 0.0 else f"{alpha:.1f}"
    print(f"{label:>8} | {auc:.6f}")

best_alpha = max(ridge_results, key=ridge_results.get)
print(f"\nOptimal alpha={best_alpha}, OOF AUC={ridge_results[best_alpha]:.6f}")

# 최종 스태킹 수행
final_oof, final_test, final_auc = make_ecdf_ridge_stack(
    model_names, mean_rank_oofs, mean_rank_tests, y, alpha=best_alpha
)

# ==================== submission 저장 ====================
submission['probability'] = final_test
output_sub_dir = os.path.join(BASE_DIR, "src", "v17")
os.makedirs(output_sub_dir, exist_ok=True)

output_sub_path = os.path.join(output_sub_dir, f"submission_v17_bag_{final_auc:.6f}.csv")
submission.to_csv(output_sub_path, index=False)
print(f"\nSaved v17 ECDF Bagged Ridge Stack Submission to: {output_sub_path}")

# 백업용으로 데이터 폴더 및 제출용 폴더에도 저장
backup_sub_path = os.path.join(DATA_DIR, f"submission_v17_bag_{final_auc:.6f}.csv")
submission.to_csv(backup_sub_path, index=False)

team_submit_dir = os.path.join(BASE_DIR, "submission file")
os.makedirs(team_submit_dir, exist_ok=True)
team_submit_path = os.path.join(team_submit_dir, f"submission_v17_bag_{final_auc:.6f}.csv")
submission.to_csv(team_submit_path, index=False)
print(f"Saved submission to team submit folder: {team_submit_path}")
