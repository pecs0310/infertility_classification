import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

import warnings
warnings.filterwarnings('ignore')

# ==================== 환경 및 경로 설정 ====================
DATA_DIR = "./data" 
if not os.path.exists(DATA_DIR):
    if os.path.exists("/kaggle/input"):
        dirs = [os.path.join("/kaggle/input", d) for d in os.listdir("/kaggle/input")]
        if dirs:
            DATA_DIR = dirs[0]
            print(f"[Kaggle Mode] Auto-detected input directory: {DATA_DIR}")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
train = pd.read_csv(TRAIN_PATH)
target_col = '임신 성공 여부'

# 진짜 결측 대치용 Train 통계치 산정
train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0

# 피처 빌딩 (train_v19_opt_gpu.py와 완전 동일)
def build_features(df, transfer_median):
    df_new = df.copy()
    
    # 3.1 특정 시술 유형 보간
    if '특정 시술 유형' in df_new.columns and '미세주입된 난자 수' in df_new.columns:
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

    # Teammate's indicators
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

    # v16 Dual missingness flags
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

    # 배아 이식 경과일 수동 대치
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median

    # Outlier Clipping
    CLIP_RULES = {
        "총 생성 배아 수": 40, "수집된 신선 난자 수": 40, "미세주입된 난자 수": 45,
        "혼합된 난자 수": 40, "저장된 배아 수": 30, "배아 이식 경과일": 7, "난자 혼합 경과일": 7,
        "배아 해동 경과일": 2, "난자 해동 경과일": 1
    }
    for col, upper in CLIP_RULES.items():
        if col in df_new.columns:
            df_new[f"{col}_high_flag"] = (df_new[col] > upper).astype(int)
            df_new[col] = df_new[col].clip(upper=upper)

    # 배아 이식 경과일 플래그
    if "배아 이식 경과일" in df_new.columns:
        df_new["transfer_day_0_1_flag"] = df_new["배아 이식 경과일"].isin([0, 1]).astype(int)
        df_new["transfer_day_3_flag"] = (df_new["배아 이식 경과일"] == 3).astype(int)
        df_new["transfer_day_5_or_more_flag"] = (df_new["배아 이식 경과일"] >= 5).astype(int)

    # 6 Ratios
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

    # 실효 모성 나이 및 격차
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

    # 나이 Ordinal 매핑 & 플래그
    age_map = {
        '만18-34세': 0, '만35-37세': 1, '만38-39세': 2, 
        '만40-42세': 3, '만43-44세': 4, '만45-50세': 5, '알 수 없음': np.nan
    }
    donor_age_map = {
        '만20세 이하': 0, '만21-25세': 1, '만26-30세': 2, 
        '만31-35세': 3, '만36-40세': 4, '만41-45세': 5, '알 수 없음': np.nan
    }
    
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

    # 시술유형 세부 토큰화
    if "특정 시술 유형" in df_new.columns:
        s = df_new["특정 시술 유형"].astype(str)
        for token in ["IVF", "ICSI", "IUI", "ICI", "GIFT", "FER", "Generic DI", "IVI", "BLASTOCYST", "AH"]:
            safe = token.lower().replace(" ", "_")
            df_new[f"spec_has_{safe}"] = s.str.contains(token, regex=False, na=False).astype(int)

    # 나이 & 난자 출처 인터랙션
    if {"시술 당시 나이", "난자 출처"}.issubset(df_new.columns):
        df_new["age_oocyte_source"] = df_new["시술 당시 나이"].astype(str) + "_" + df_new["난자 출처"].astype(str)

    # 시술/임신/출산 횟수 변환
    count_map = {'0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6}
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        if col in df_new.columns:
            df_new[f'{col}_int'] = df_new[col].map(count_map)

    # 배아 배양 일수 계산
    if "배아 이식 경과일" in df_new.columns and "난자 혼합 경과일" in df_new.columns:
        df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']

    # 남성/여성 불임 요인 플래그
    male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
    male_present = [f for f in male_factors if f in df_new.columns]
    if male_present:
        df_new['is_male_infertility'] = df_new[male_present].any(axis=1).astype(int)
        
    female_factors = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '여성 주 불임 원인', '여성 부 불임 원인']
    female_present = [f for f in female_factors if f in df_new.columns]
    if female_present:
        df_new['is_female_infertility'] = df_new[female_present].any(axis=1).astype(int)

    # 원래 나이 및 횟수 제거
    cols_to_drop = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + count_cols
    df_new = df_new.drop(columns=[c for c in cols_to_drop if c in df_new.columns], errors='ignore')
    
    # Dead Columns 제거
    DEAD_COLS = ["불임 원인 - 여성 요인", "난자 채취 경과일"]
    df_new = df_new.drop(columns=[c for c in DEAD_COLS if c in df_new.columns], errors='ignore')

    return df_new

X = build_features(train, train_transfer_median)
features = [col for col in X.columns if col not in ['ID', target_col]]
y = train[target_col].values.astype(int)
X = X[features].copy()

# 범주형 처리
SENTINEL = "정보없음"
cat_cols = []
for col in features:
    if not pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
        cat_cols.append(col)
        X[col] = X[col].astype(str).replace('nan', np.nan).fillna(SENTINEL)
        train_categories = sorted(list(set(X[col].unique()) | {SENTINEL}))
        X[col] = pd.Categorical(X[col], categories=train_categories)

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# GPU 환경 활성화 여부 확인
HAS_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        HAS_GPU = True
except:
    pass

# Optuna Objective
def objective(trial):
    # 하이퍼파라미터 탐색 범위 지정
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 1200,
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.06, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'allow_writing_files': False,
        'verbose': False,
        'task_type': 'GPU' if HAS_GPU else 'CPU'
    }
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for tr_idx, val_idx in folds.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        
        tp = Pool(X_tr, y_tr, cat_features=cat_idx)
        vp = Pool(X_va, y_va, cat_features=cat_idx)
        
        model = CatBoostClassifier(**params)
        model.fit(tp, eval_set=vp, use_best_model=True)
        oof[val_idx] = model.predict_proba(vp)[:, 1]
        
    score = roc_auc_score(y, oof)
    return score

if __name__ == "__main__":
    print(f"Optuna CatBoost GPU Tuning Mode | GPU Available: {HAS_GPU}")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("\n" + "=" * 50)
    print("Optimization Completed!")
    print(f"Best Trial Score (OOF AUC): {study.best_value:.6f}")
    print("Best Trial Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)
