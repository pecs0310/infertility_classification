"""
난임 시술 임신 성공 여부 예측 - LightGBM Baseline
=================================================

데이터: 난임 시술(IVF/DI) 환자 정보 -> 임신 성공 여부(0/1) 예측
평가지표: ROC-AUC (확률값 기준)

파이프라인 구성
1. 데이터 로드
2. 전처리 (ID 제거, 결측치 처리, 라벨 인코딩)
3. 학습/검증 분할 + 모델 학습
4. 검증 성능 평가 (ROC-AUC)
5. 테스트 예측 + 제출 파일 생성

폴더 구조 가정:
    infertility_classification/   (이 리포)
    ├── main.py
    └── data/
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv

실행:
    python main.py
"""

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "result.csv")

TARGET_COL = "임신 성공 여부"
ID_COL = "ID"
RANDOM_STATE = 1004
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"[로드] train: {train.shape}, test: {test.shape}")
    return train, test


# ---------------------------------------------------------------------------
# 2. 전처리
# ---------------------------------------------------------------------------
def drop_id(train: pd.DataFrame, test: pd.DataFrame):
    """ID 컬럼은 학습에 쓰지 않으므로 제거. test의 ID는 제출 파일 작성에 필요하므로 따로 보관."""
    test_ids = test[ID_COL].copy()
    train = train.drop(columns=[ID_COL])
    test = test.drop(columns=[ID_COL])
    return train, test, test_ids


def fill_missing_values(train: pd.DataFrame, test: pd.DataFrame):
    """
    결측치 처리
    - 범주형(object/str) 컬럼: train 기준 최빈값으로 대체
    - 수치형(int/float) 컬럼: train 기준 중앙값으로 대체
    train에서 계산한 값을 test에도 동일하게 적용 (데이터 누수 방지)
    """
    categorical_cols = train.select_dtypes(include=["object", "str"]).columns
    numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns

    na_cols = train.isna().sum().loc[lambda x: x > 0].index

    categorical_na_cols = [c for c in na_cols if c in categorical_cols]
    numerical_na_cols = [c for c in na_cols if c in numerical_cols]

    for col in categorical_na_cols:
        most_frequent = train[col].mode()[0]
        train[col] = train[col].fillna(most_frequent)
        test[col] = test[col].fillna(most_frequent)

    for col in numerical_na_cols:
        median_value = train[col].median()
        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)

    print(f"[결측치 처리] 범주형 {len(categorical_na_cols)}개, 수치형 {len(numerical_na_cols)}개 컬럼 처리 완료")
    return train, test


def encode_categorical(train: pd.DataFrame, test: pd.DataFrame):
    """
    범주형 변수를 LabelEncoder로 인코딩.
    test에만 등장하는 새로운 라벨이 있으면 인코더의 classes_에 추가해서 처리.
    """
    categorical_cols = train.select_dtypes(include=["object", "str"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train[col])

        # test에만 존재하는 라벨을 classes_에 추가
        unseen_labels = set(test[col].unique()) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.append(le.classes_, list(unseen_labels))

        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    print(f"[인코딩] 범주형 컬럼 {len(categorical_cols)}개 라벨 인코딩 완료")
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train, test, test_ids = drop_id(train, test)
    train, test = fill_missing_values(train, test)
    train, test = encode_categorical(train, test)
    return train, test, test_ids


# ---------------------------------------------------------------------------
# 3. 학습
# ---------------------------------------------------------------------------
def train_model(train: pd.DataFrame):
    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = LGBMClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_val, y_val


# ---------------------------------------------------------------------------
# 4. 평가
# ---------------------------------------------------------------------------
def evaluate(model, X_val: pd.DataFrame, y_val: pd.Series):
    # ROC-AUC는 확률값으로 계산해야 함 (0/1 예측값을 넣으면 점수가 왜곡됨)
    val_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_proba)
    print(f"[검증] ROC-AUC Score: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# 5. 예측 및 제출 파일 생성
# ---------------------------------------------------------------------------
def predict_and_submit(model, test: pd.DataFrame, test_ids: pd.Series):
    test_proba = model.predict_proba(test)[:, 1]

    submission = pd.read_csv(SUBMISSION_PATH)
    submission[ID_COL] = test_ids.values
    submission["probability"] = test_proba

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] 제출 파일 생성 완료 -> {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------
def main():
    train, test = load_data()
    train, test, test_ids = preprocess(train, test)

    model, X_val, y_val = train_model(train)
    evaluate(model, X_val, y_val)

    predict_and_submit(model, test, test_ids)


if __name__ == "__main__":
    main()
