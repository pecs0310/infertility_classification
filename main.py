"""
난임 시술 임신 성공 여부 예측 - LightGBM Baseline
=================================================

데이터: 난임 시술(IVF/DI) 환자 정보 -> 임신 성공 여부(0/1) 예측
평가지표: ROC-AUC (확률값 기준)

파이프라인 구성
1. 데이터 로드
2. 전처리 (ID 제거, 구조적 결측 플래그 추가, 무의미 컬럼 제거, 결측치 처리, 라벨 인코딩)
3. Stratified 5-Fold 학습 (각 fold마다 모델 1개씩, 총 5개 모델)
4. 검증 성능 평가 (Fold별 ROC-AUC + 평균/표준편차)
5. 테스트 예측 (5개 모델 예측 확률 평균) + 제출 파일 생성

폴더 구조 가정:
    infertility_classification/
    ├── pyproject.toml
    ├── main.py
    └── data/
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv

실행 환경: uv로 관리되는 가상환경 (.venv)
필요 패키지: pandas, numpy, scikit-learn, lightgbm (pyproject.toml에 등록됨)

최초 설정 (한 번만):
    uv add pandas numpy scikit-learn lightgbm jupyter ipykernel openpyxl

실행:
    uv run main.py
    (또는 .venv 활성화 후 python main.py)
"""

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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


def add_structural_flags(train: pd.DataFrame, test: pd.DataFrame):
    """
    구조적 결측을 신호로 활용하기 위한 플래그 컬럼 추가.

    데이터 탐색 결과, 배아/난자 수치 관련 컬럼들의 결측(6,291건)은
    전부 '시술 유형 = DI'(인공수정)와 정확히 일치한다. DI는 체외에서
    배아를 만들지 않으므로 이 결측은 '정보 없음'이 아니라 '해당없음'이다.
    또한 배아 해동 경과일 등은 '동결 배아 사용 여부=0'일 때 결측되는
    단계 조건부 결측이다.

    이 두 가지를 압축된 플래그 컬럼으로 명시해주면, 모델이 중앙값으로
    채워진 값과 '원래 결측이었던' 값을 구분할 수 있게 된다.
    """
    for df in (train, test):
        df["is_DI"] = (df["시술 유형"] == "DI").astype(int)
        df["froze_embryo"] = df["동결 배아 사용 여부"].fillna(0).astype(int)
    return train, test


def fill_missing_values(train: pd.DataFrame, test: pd.DataFrame):
    """
    결측치 처리
    - 범주형 컬럼: "해당없음" 카테고리로 명시적 대체
    - 수치형 컬럼: -1 (실제 데이터 범위 밖의 특수값)로 대체

    단순 최빈값/중앙값 대체는 'DI라서 배아 수치가 없는 것'과
    '실제로 측정이 안 된 것'을 구분하지 못한다. 대부분의 결측이
    무작위가 아니라 구조적 결측(DI 시술 전체, 또는 특정 단계 미실시)
    이므로, 실제 관측값과 섞이지 않는 특수값으로 명시적으로 표시한다.
    add_structural_flags()에서 추가한 플래그와 함께 쓰여
    모델이 결측 자체를 신호로 학습할 수 있게 한다.
    """
    categorical_cols = train.select_dtypes(include=["object", "str"]).columns
    numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns

    na_cols = train.isna().sum().loc[lambda x: x > 0].index

    categorical_na_cols = [c for c in na_cols if c in categorical_cols]
    numerical_na_cols = [c for c in na_cols if c in numerical_cols]

    for col in categorical_na_cols:
        train[col] = train[col].fillna("해당없음")
        test[col] = test[col].fillna("해당없음")

    for col in numerical_na_cols:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)

    print(f"[결측치 처리] 범주형 {len(categorical_na_cols)}개 -> '해당없음', 수치형 {len(numerical_na_cols)}개 -> -1 대체 완료")
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


def drop_uninformative_columns(train: pd.DataFrame, test: pd.DataFrame):
    """
    정보량이 거의 없는 컬럼 제거.

    - 불임 원인 - 여성 요인: 전체 데이터에서 값이 전부 0 (분산 0, 학습에 기여 불가)
    - 난자 채취 경과일: 관측값이 거의 전부 0이고 나머지는 결측이라 분산이 거의 없음

    5-Fold 교차검증으로 확인한 결과, 제거 시 ROC-AUC가 일관되게(여러 random_state에서)
    소폭 개선됨 (+0.00002 ~ +0.00013). 효과는 작지만 의미 없는 컬럼을 남겨둘 이유가 없어 제거.
    """
    cols_to_drop = ["불임 원인 - 여성 요인", "난자 채취 경과일"]
    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)
    print(f"[컬럼 제거] 정보량 없는 컬럼 {len(cols_to_drop)}개 제거: {cols_to_drop}")
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train, test, test_ids = drop_id(train, test)
    train, test = add_structural_flags(train, test)
    train, test = drop_uninformative_columns(train, test)
    train, test = fill_missing_values(train, test)
    train, test = encode_categorical(train, test)
    return train, test, test_ids


# ---------------------------------------------------------------------------
# 3. 학습 (K-Fold 앙상블)
# ---------------------------------------------------------------------------
N_SPLITS = 5


def train_kfold(train: pd.DataFrame):
    """
    Stratified K-Fold로 N_SPLITS개 모델을 학습.

    단일 train_test_split은 검증 점수가 어떤 데이터가 검증셋으로 빠지느냐에
    따라 운에 좌우될 수 있다 (실험 중 random_state를 바꾸면 점수가
    0.737~0.745 사이로 흔들리는 것을 확인). K-Fold는 전체 데이터를
    N_SPLITS번 나눠 번갈아 검증셋으로 사용하므로:
    - 검증 점수가 더 안정적 (모든 데이터가 한 번씩 검증에 사용됨)
    - N_SPLITS개 모델의 예측을 평균내는 앙상블 효과로 test 예측이 더 안정적
    """
    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_proba)
        fold_scores.append(score)
        models.append(model)
        print(f"[Fold {fold}/{N_SPLITS}] ROC-AUC: {score:.4f}")

    return models, fold_scores


# ---------------------------------------------------------------------------
# 4. 평가
# ---------------------------------------------------------------------------
def evaluate(fold_scores: list):
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"[검증] {N_SPLITS}-Fold 평균 ROC-AUC: {mean_score:.4f} (std: {std_score:.4f})")
    return mean_score


# ---------------------------------------------------------------------------
# 5. 예측 및 제출 파일 생성
# ---------------------------------------------------------------------------
def predict_and_submit(models: list, test: pd.DataFrame, test_ids: pd.Series):
    """N_SPLITS개 모델의 예측 확률을 평균내어 최종 제출 파일 생성 (앙상블)."""
    test_proba = np.mean([model.predict_proba(test)[:, 1] for model in models], axis=0)

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

    models, fold_scores = train_kfold(train)
    evaluate(fold_scores)

    predict_and_submit(models, test, test_ids)


if __name__ == "__main__":
    main()