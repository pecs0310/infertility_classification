import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

train = pd.read_csv(TRAIN_PATH)

train_transfer_median = train.loc[train['이식된 배아 수'] > 0, '배아 이식 경과일'].median()
if pd.isna(train_transfer_median):
    train_transfer_median = 3.0

def preprocess_data(df, transfer_median):
    df_new = df.copy()
    df_new.loc[(df_new['특정 시술 유형'] == 'Unknown') & (df_new['미세주입된 난자 수'] > 0), '특정 시술 유형'] = 'ICSI'
    
    temp_transferred = df_new['이식된 배아 수'].fillna(0)
    df_new.loc[(temp_transferred > 0) & (df_new['배아 이식 경과일'].isna()), '배아 이식 경과일'] = transfer_median
    df_new['배아 이식 경과일'] = df_new['배아 이식 경과일'].fillna(-1)
    
    df_new.loc[(df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이'].isna() | (df_new['난자 기증자 나이'] == '알 수 없음')), '난자 기증자 나이'] = '만31-35세'
    df_new.loc[(df_new['정자 출처'] == '기증 제공') & (df_new['정자 기증자 나이'].isna() | (df_new['정자 기증자 나이'] == '알 수 없음')), '정자 기증자 나이'] = '만21-25세'
    
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
    
    df_new['is_advanced_age'] = (df_new['시술 당시 나이_ordinal'] >= 1).astype(float)
    df_new['is_elderly_age'] = (df_new['시술 당시 나이_ordinal'] >= 3).astype(float)
    
    count_map = {'0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6}
    count_cols = [
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', 
        '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수'
    ]
    for col in count_cols:
        df_new[f'{col}_int'] = df_new[col].map(count_map)
        
    df_new['elderly_self_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '본인 제공')).astype(float)
    df_new['elderly_donor_egg'] = ((df_new['is_elderly_age'] == 1) & (df_new['난자 출처'] == '기증 제공')).astype(float)
    df_new['young_donor_egg'] = ((df_new['난자 출처'] == '기증 제공') & (df_new['난자 기증자 나이_ordinal'] <= 2)).astype(float)
    
    df_new['embryo_culture_days'] = df_new['배아 이식 경과일'] - df_new['난자 혼합 경과일']
    
    male_factors = ['불임 원인 - 남성 요인', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태', '남성 주 불임 원인', '남성 부 불임 원인']
    df_new['is_male_infertility'] = df_new[male_factors].any(axis=1).astype(int)
    
    female_factors = ['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증', '여성 주 불임 원인', '여성 부 불임 원인']
    df_new['is_female_infertility'] = df_new[female_factors].any(axis=1).astype(int)
    
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
    
    cols_to_drop = ['시술 당시 나이', '난자 기증자 나이', '정자 기증자 나이'] + count_cols
    df_new = df_new.drop(columns=cols_to_drop)
    
    return df_new

X = preprocess_data(train, train_transfer_median)
target_col = '임신 성공 여부'
drop_cols = ['ID', target_col] if target_col in X.columns else ['ID']
features = [col for col in X.columns if col not in drop_cols]
print(f"Number of final features: {len(features)}")
print("Final features list:")
for idx, col in enumerate(features):
    print(f"{idx+1}: {col}")
