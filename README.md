# Infertility_classification_model

## Background

난임은 전 세계적으로 증가하는 중요한 의료 문제로, 많은 부부들이 오랜 기간 동안 신체적·정신적 부담을 겪고 있습니다. 난임 시술을 진행하는 환자들은 치료 과정에서 높은 비용과 심리적 스트레스를 경험하기 때문에, 최소한의 시술로 임신 성공 가능성을 높이는 것이 매우 중요합니다.

이러한 요구에 부응하기 위해 의료기관들은 인공지능(AI)을 활용한 임신 성공 여부 예측 모델에 큰 관심을 보이고 있습니다. AI 기반 솔루션은 방대한 난임 치료 데이터를 분석하여 최적의 의사결정을 지원하고, 환자 맞춤형 치료 계획을 수립하는 데 기여할 수 있습니다. 이는 환자의 시술 부담을 줄이는 동시에, 의료기관이 차별화된 서비스를 제공할 수 있도록 돕는 중요한 경쟁 요소가 될 것입니다.

## Goal

난임 환자 대상 임신 성공 여부 예측 AI모델 개발

- 난임 환자 데이터를 분석하여 임신 성공에 영향을 미치는 주요 요인을 도출하고, 정확한 예측을 위한 최적의 AI 모델을 개발해야 합니다.
  ※ 임신 성공: 출산까지 성공적으로 진행된 임신

## Rules

1. 평가산식 : ROC-AUC
   Public에서 Test 데이터 100%를 활용하여 평가

2. 외부 데이터 및 사전 학습 모델
   외부 데이터 사용 불가 (사전 학습 모델(Pre-trained Model) 사용 가능)

3. 유의 사항
   **1일 최대 제출 횟수: 3회**
   모델 학습에서 평가 데이터셋 활용(Data Leakage)시 수상 제외 - label encoding, one-hot encoding 시 test 데이터 셋 활용 - data scaling 적용 시 test 데이터 셋 활용 - test 데이터 셋에 pd.get_dummies() 함수 적용 - test 데이터 셋의 결측치 처리 시 test 데이터 셋의 통계 값 활용 - 위 예시 외에도 test 데이터 셋이 모델 학습에 활용되는 경우에 Data leakage에 해당됨

<br>
<br>
<br>

---

## 팀 구성

| 이름   | 담당 기능                 |
| ------ | ------------------------- |
| 박소정 | Everybody does everything |
| 박지은 | Everybody does everything |
| 김영혜 | Everybody does everything |

---

## 프로젝트 진행 과정

### 1. Team Rule 정의

- **브랜치 전략**: GitHub Flow 사용
  - `main` 브랜치를 기준으로 `feature/조원명` 브랜치를 생성하여 작업
  - 작업완료 후 `develop` 브랜치로 PR (선택적)
  - 이후 PR을 통해 1인이상에게 리뷰를 받아 `main` 브랜치로 머지

- **커밋 컨벤션**:
  - `feat`: 새로운 기능 추가
  - `fix`: 버그 수정
  - `docs`: 문서 작성/수정
  - `chore`: 설정 변경
- **코드 리뷰**: PR 생성 후 팀원 검토 후 머지

### 2. [Data 명세](data/데이터%20명세.xlsx)

### 4. Git & GitHub Branch 전략

**Git~GitHub Flow** 채택:

```
main
 ├── develop                # main merge 전 대기코드
 ├── feature/sojung         # 박소정 브랜치
 ├── feature/yunghye        # 김영혜 브랜치
 └── feature/genie          # 박지은 브랜치
```

### 5. 프로젝트 세팅

**기술 스택:**

| 구분 | 기술 |
| ---- | ---- |
| TBD  | TBD  |

**환경 설정:**

```bash
# 라이브러리 설치
uv add ...
```

## 프로젝트 구조

```
root/
├── data/               # 대화제공데이터 및 문서
│   ├── 데이터 명세.xlsx   # 데이터항목 설명
│   ├── sample_submission.csv    # 결과제출용 템플릿
│   ├── test.csv        # 추론용 데이터
│   └── train.csv       # 훈련용 데이터
├── main.py
├── .gitignore
└── README.md          # 프로젝트개요
```
