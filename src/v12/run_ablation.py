import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

target_col = '임신 성공 여부'
train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0

def preprocess_ablation(df, transfer_median, step):
    df_new = df.copy()
    
    # 1. Manual Imputation (Step 3에서 제외됨)
    if step != 3:
        # 특정 시술 유형 보간
        df_new.loc[(df_new['특정 시술 유형'] == 'Unknown') & (df_new['미세주입된 난자 수'] > 0), '특정 시술 유형'] = 'ICSI'
        # 배아 이식 경과일 수동 대치
        temp_transferred = df_new['이식된 배아 수'].fillna(0)
        df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median
        df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)
        # 난자 및 정자 기증자 나이 결측치 대치
        df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
        df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'
    else:
        # Step 3: 수동 대치 없이 단순 fillna만 적용
        df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)

    # 나이 Ordinal 매핑
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
    
    count_map = {'0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6}
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        df_new[f'{col}_int'] = df_new[col].map(count_map)

    # 2. Rejuvenation Features (1등 솔루션 기법, 공통 적용)
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

    # 3. Custom Ratios (Step 0 에서만 유지)
    if step == 0:
        df_new['pregnancy_efficiency'] = df_new['총 임신 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
        df_new['delivery_efficiency'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
        df_new['previous_success_rate'] = df_new['총 출산 횟수_int'] / (df_new['총 시술 횟수_int'] + 1)
        df_new['egg_to_embryo_ratio'] = df_new['총 생성 배아 수'] / (df_new['수집된 신선 난자 수'] + 1)
        df_new['embryo_stored_ratio'] = df_new['저장된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
        df_new['embryo_transferred_ratio'] = df_new['이식된 배아 수'] / (df_new['총 생성 배아 수'] + 1)
        df_new['icsi_fertilization_rate'] = df_new['미세주입에서 생성된 배아 수'] / (df_new['미세주입된 난자 수'] + 1)
        df_new['embryo_transfer_efficiency'] = df_new['이식된 배아 수'] / (df_new['저장된 배아 수'] + 1)

    # 4. is_missing Flags (Step 0, Step 1 에서만 유지)
    if step in [0, 1]:
        df_new['is_missing_eggs'] = df_new['수집된 신선 난자 수'].isna().astype(int)
        df_new['is_missing_embryos'] = df_new['총 생성 배아 수'].isna().astype(int)
        df_new['is_missing_transfer'] = df_new['배아 이식 경과일'].isna().astype(int)
        df_new['is_transfer_missing'] = ((df_new['배아 이식 경과일'].isna()) | (df_new['이식된 배아 수'] == 0)).astype(int)
        df_new['frozen_only_cycle'] = ((df_new['배아 이식 경과일'].isna()) & (df_new['저장된 배아 수'] > 0)).astype(int)
        df_new['infertility_severity_score'] = df_new[[
            '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', 
            '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', 
            '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'
        ]].sum(axis=1)

    # 5. Custom Flags (Step 5에서 제외됨)
    if step != 5:
        df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(float)
        df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(float)
        df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(float)
        df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(float)
        df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(float)
        df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']
        
        male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
        df_new['is_male_infertility'] = df_new[male_factors].any(axis=1).astype(int)
        
        female_factors = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '여성 주 불임 원인', '여성 부 불임 원인']
        df_new['is_female_infertility'] = df_new[female_factors].any(axis=1).astype(int)

    # 6. High-Null Columns (Step 4에서 제외됨)
    if step == 4:
        high_null_cols = ['PGS 시술 여부', 'PGD 시술 여부', '착상 전 유전 검사 사용 여부', '착상 전 유전 진단 사용 여부']
        high_null_cols = [c for c in high_null_cols if c in df_new.columns]
        df_new = df_new.drop(columns=high_null_cols)

    # 원래 나이 및 횟수 문자열 컬럼 드롭
    cols_to_drop = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + count_cols
    df_new = df_new.drop(columns=cols_to_drop)
    
    return df_new

def run_ablation_step(step_num, step_name):
    print(f"\n==================== RUNNING ABLATION STEP {step_num}: {step_name} ====================")
    
    # 데이터 전처리
    X_step = preprocess_ablation(train, train_transfer_median, step_num)
    
    drop_cols = ['ID', target_col] if target_col in X_step.columns else ['ID']
    features = [col for col in X_step.columns if col not in drop_cols]
    y_step = train[target_col]
    X_step = X_step[features].copy()
    
    # 범주형 처리
    cat_cols = []
    for col in features:
        if not pd.api.types.is_numeric_dtype(X_step[col]) and not pd.api.types.is_bool_dtype(X_step[col]):
            cat_cols.append(col)
            X_step[col] = X_step[col].astype(str).replace('nan', np.nan)
            unique_cats = sorted(list(X_step[col].dropna().unique()))
            X_step[col] = pd.Categorical(X_step[col], categories=unique_cats)
            X_step[col] = X_step[col].cat.codes.astype('category')
            
    print(f"Number of features: {len(features)}")
    
    # 5-Fold Stratified CV
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_step))
    
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
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(X_step, y_step)):
        X_tr, y_tr = X_step.iloc[train_idx], y_step.iloc[train_idx]
        X_va, y_va = X_step.iloc[val_idx], y_step.iloc[val_idx]
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        
        callbacks = [lgb.early_stopping(50, verbose=False)]
        model = lgb.train(
            lgb_params, trn_data, num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=callbacks
        )
        oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        
    score = roc_auc_score(y_step, oof)
    print(f"Step {step_num} CV AUC: {score:.6f}")
    return len(features), score

if __name__ == "__main__":
    steps = [
        (0, "Base Config (v11-like)"),
        (1, "Remove Custom Ratios"),
        (2, "Remove is_missing Flags (Optimal v12)"),
        (3, "Revert Manual Imputations"),
        (4, "Remove 4 High-Null Columns"),
        (5, "Remove Custom Flags")
    ]
    
    results = []
    base_auc = 0.0
    for num, name in steps:
        feat_cnt, score = run_ablation_step(num, name)
        if num == 0:
            base_auc = score
            diff = 0.0
        else:
            diff = score - base_auc
        results.append({
            'Step': f"Step {num}: {name}",
            'Features': feat_cnt,
            'CV_AUC': score,
            'Diff': diff
        })
        
    df_results = pd.DataFrame(results)
    print("\n==================== ABLATION STUDY SUMMARY ====================")
    print(df_results.to_string(index=False))
    
    os.makedirs(os.path.join(BASE_DIR, "scratch"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "scratch", "ablation_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")
