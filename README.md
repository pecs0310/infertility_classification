# AI Health Web Assignment

흉부 X-Ray 이미지를 활용한 폐렴 판독 백오피스 시스템입니다.

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
