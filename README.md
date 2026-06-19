# 👶 난임 환자 대상 임신 성공 여부 예측 AI 모델 개발

## 📖 프로젝트 배경 및 목표
난임은 전 세계적으로 증가하는 중요한 의료 문제로, 많은 부부들이 신체적·정신적 부담을 겪고 있습니다. 본 프로젝트는 난임 환자들의 시술 데이터를 분석하여 **'임신 성공 여부'를 예측하는 최적의 AI 모델**을 개발하는 것을 목표로 합니다. 
이를 통해 최소한의 시술로 임신 성공 가능성을 높이고, 환자 맞춤형 치료 계획 수립을 지원하고자 합니다.

- **주최/주관:** DACON (데이콘)
- **평가 지표:** ROC-AUC
- **개발 환경:** Python, Jupyter Notebook, LightGBM/XGBoost 등

---

## 📊 데이터셋 정보 (Dataset)
* **Train Data:** 256,351 rows (67개 환자 및 시술 특성 컬럼)
* **Test Data:** 90,067 rows
* **Target Variable:** 임신 성공 여부 (`1`: 임신 성공(출산), `0`: 임신 실패)
* *주의: 본 대회 규정에 따라 원본 데이터(`train.csv`, `test.csv`)는 Github에 업로드하지 않습니다.*

---

## 📁 디렉토리 구조 (Directory Structure)
```text
infertility_classification/
│
├── data/               # 데이터 폴더 (gitignore 처리됨)
│   ├── train.csv
│   └── test.csv
│
├── notebooks/          # EDA 및 실험용 Jupyter Notebook
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
│
├── src/                # 모듈화된 파이썬 스크립트
│   ├── preprocess.py   # 전처리 코드
│   └── train.py        # 모델 학습 코드
│
├── .gitignore          # 깃허브 업로드 제외 목록
├── README.md           # 프로젝트 설명서
└── requirements.txt    # 필요한 파이썬 라이브러리 목록