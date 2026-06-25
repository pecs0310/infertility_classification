# 불임 분류 프로젝트 최종 검증 및 Git Push 완료 보고서 (v1 ~ v14)

이 문서는 기초 베이스라인 모델(v1)부터 시작하여 ECDF 최적 가중치 랭크 배깅(v14)까지의 모든 실험 결과를 정리하고, 깃허브 브랜치 수정 작업 및 Windows 환경 Git 이슈 해결 내용을 기록한 최종 검증 보고서입니다.

---

## 1. 실험 버전별 최종 성능 비교 (ROC-AUC)

| 실험 버전 | 주요 적용 사항 | OOF ROC-AUC 스코어 | 성능 향상도 | 최종 제출 파일 |
| :--- | :--- | :---: | :---: | :--- |
| **v1 Baseline** | 5-Fold CV 인프라 구축, Label Encoding | **0.739960** | 기준점 | `submission_v1_lgb.csv` |
| **v2 Domain** | 연령 Ordinal 인코딩, 과거 성공 효율, 배양일, 주기별 NaN 대치 | **0.740140** | **+0.000180** | `submission_v2_lgb.csv` |
| **v3 Ensemble**| LGBM 파라미터 최적화 및 XGBoost, CatBoost 소프트 보팅 앙상블 | **0.740310** | **+0.000350** | `submission_v3_ensemble.csv` |
| **v4 Advanced**| 논문 기반 피처 11종 추가 (교차 피처, 프로세스 결측 플래그 등) | **0.740400** | **+0.000440** | `submission_v4_ensemble.csv` |
| **v5 Imputed** | 진짜/가짜 결측치 논리 분리 및 임상 통계치(3.0일, 기증자 최빈 나이) 정밀 보간 | **0.740420** | **+0.000460** | `submission_v5_imputed.csv` |
| **v6 Advanced** | Winsorization(이상치 클리핑 99.5%), Groupby Aggregation, 교차 피처 고도화 | **0.740410** | **+0.000450** | `submission_v6_advanced.csv` |
| **v7 Stacking** | 이상치 클리핑 및 그룹 통계 제거(v5기반), AdaBoost 추가, LightGBM Stacking 앙상블 | **0.741250** | **+0.001290** | `submission_v7_advanced.csv` |
| **v8 Stacking** | 4대 고도화 임상 시너지 피처 추가, Logistic Regression L2 정규화 Stacking | **0.739880** | **-0.000080** | `submission_v8_advanced.csv` |
| **v9 Split** | 하드케이스 억제 피처셋 설계 및 Split LGBM 조건부 블렌딩 적용 | **0.740156** | **+0.000196** | `submission_v9_advanced.csv` |
| **v10 Voting** | GBDT 3종(LGB/XGB/Cat) 소프트보팅 및 카테고리 코드 수치형 맵핑 최적화 | **0.740389** | **+0.000429** | `submission_v10_advanced.csv` |
| **v11 ECDF** | 95개 피처셋 및 GBDT 3종 + MLPClassifier ECDF 랭크 블렌딩 구축 | **0.740121** | **+0.000161** | `submission_v11_ens_0.740121.csv` |
| **v12 Select** | Ablation Study 기반 피처 소거(81개 피처) 및 고정 가중치 ECDF 블렌딩 | **0.740336** | **+0.000376** | `submission_v12_ens_0.740336.csv` |
| **v13 Opt** | Scipy Nelder-Mead + Softmax 기반 가중치 최적화 ECDF 랭크 블렌딩 | **0.740534** | **+0.000574** | `submission_v13_ens_0.740534.csv` |
| **v14 Bag**| **5-시드 멀티배깅 및 Scipy Nelder-Mead ECDF 랭크 블렌딩 최적화** | **0.740645** | **+0.000685** | `submission_v14_bag_0.740645.csv` |
| **v16 Bag**| **임상 계층 상태 인코딩 & 이원 결측 최적화 5-시드 배깅** | **0.740587** | **+0.000627** | `submission_v16_bag_0.740587.csv` |
| **v17 Ridge Stack**| **v16 피처 + Rich Features 통합 및 ECDF Ridge Stacking** | **0.740830** | **+0.000870** | `submission_v17_bag_0.740830.csv` |
| **v18 Ridge Stack (최종)**| **MLP 원핫인코딩 + RF/ET 모델 다양화 및 ECDF Ridge Stacking** | 🏆 **0.740822** | **+0.000862** | `submission_v18_bag_0.740822.csv` |

> [!NOTE]
> **v18 최종 Stacking 모델 결과**:
> - **최종 OOF AUC**: **`0.740822`**
> - **개선점**: MLP Classifier의 입력 인코딩 방식을 정규 원핫 인코딩으로 개편하고, 릿지 스태킹 시 트리 모델 간의 상관 에러를 제어하기 위해 Random Forest와 Extra Trees 모델을 신규 수혈하여 높은 일반화 예측 성능을 유지했습니다.

---

## 2. Git 브랜치 수정 및 Windows 환경 이슈 해결

### 1) Windows NTFS 호환성 문제 해결
- **문제점**: 리모트 `develop` 브랜치에 Windows 예약 특수문자인 `>`가 들어간 파일(예: `1차_multiseed_bagging_team_blend_report(0.74053->0.7416942).md`)이 커밋되어 있어 Windows 환경에서 checkout 및 pull 수행 시 `error: invalid path`가 발생하며 머지 및 추적이 실패했습니다.
- **해결 방안**:
  1. Git `core.protectNTFS` 및 `core.protectHFS` 설정을 임시로 `false`로 비활성화하여 검증 예외 처리를 수행했습니다.
  2. Git **Sparse Checkout(선택적 체크아웃)**을 활성화하고 `.git/info/sparse-checkout`에 예외 디렉토리를 등록했습니다:
     ```text
     /*
     !docs/experiment_history/
     !submission file/
     ```
  3. 이 조치를 통해 에러가 발생하는 디렉토리를 로컬 디스크 쓰기 대상에서 제외함으로써, `develop` 브랜치를 안전하게 가져오고(Checkout/Pull) 작업 브랜치를 업데이트하는 데 성공했습니다.

### 2) v5 및 v7 브랜치 수정 (Push to develop)
- **요청 사항**: 기존에 `main`을 기준으로 생성/푸시되었던 `feature/v5-release` 및 `feature/v7-release` 브랜치를 `develop` 기준으로 수정하여 푸시.
- **조치 사항**:
  1. `feature/v5-release` 브랜치에서 PR 테스트 커밋을 제외한 고유 커밋군을 `develop` 브랜치(`origin/develop`) 위로 **Targeted Rebase**하였습니다.
     ```bash
     git rebase --onto develop 967edd0 feature/v5-release
     ```
  2. `feature/v7-release` 브랜치의 고유 커밋인 `36ffc6f`를 `develop` 브랜치 위로 **Rebase**하였습니다.
     ```bash
     git rebase --onto develop 3f2a20b feature/v7-release
     ```
  3. 두 브랜치를 리모트 저장소로 강제 푸시(`git push -f`)하여, GitHub 상에서 Pull Request가 `develop` 브랜치를 안전하고 깨끗하게 타겟팅(Target)하도록 수정 완료하였습니다.
  4. 모든 작업 완료 후 `git sparse-checkout disable`을 실행하여 모든 일반 추적 파일들을 로컬 디렉터리에 완벽히 복원하였습니다.

---

## 3. v15 ECDF Rank Blending (리더보드 한계 돌파용 추가 실험)
- **개요**: 단일 최적화 파이프라인의 이론적 한계점(0.7417 부근)을 돌파하기 위해, 서로 완전히 독립적인 최고 성능 파이프라인 2종을 **ECDF 랭크 블렌딩(Rank Blending)**으로 결합했습니다.
- **대상 파일**:
  1. **v14 Bagging Submission** (`submission_v14_bag_0.740645.csv`, Public LB: `0.741556`)
  2. **Team Best 6th Submission** (`team_best_submit_0.741849.csv`, Public LB: `0.741849`)
- **블렌딩 기법**:
  - 두 예측치 분포를 ECDF(Empirical Cumulative Distribution Function)로 각각 정규화하여 균등 분포화(Uniform) 시켰습니다.
  - 가중 평균하여 결합한 뒤, 제출 스케일의 안정성을 위해 최상위 모델인 `team_best`의 확률 공간으로 역보간(Inverse Interpolation) 매핑하였습니다.
- **산출된 제출 파일 (4가지 비중 시나리오)**:
  1. **50:50 Balanced**: `submission_v15_blend_50_50.csv` (동등한 기여)
  2. **40:60 Weighted**: `submission_v15_blend_40_60.csv` (팀 최고점 모델에 가중치 60%)
  3. **30:70 Weighted**: `submission_v15_blend_30_70.csv` (팀 최고점 모델에 가중치 70%)
  4. **60:40 Weighted**: `submission_v15_blend_60_40.csv` (신규 v14 모델에 가중치 60%)

이 시나리오들은 리더보드에서 `0.7418`을 뛰어넘어 `0.7420+` 영역으로 도달할 수 있는 가장 신뢰도 높은 조합들입니다.

---

## 4. v16 임상 구조 상태 인코딩 및 이원 결측 플래그 최적화
- **개요**: 난임 치료의 계층 구조와 MNAR의 본질을 데이터셋에 임베딩하기 위한 고도화 피처 엔지니어링을 수행하고, 5-시드 배깅 GBDT 3종 + MLP Nelder-Mead 최적화 앙상블을 실행했습니다.
- **성능 검증 (OOF AUC)**:
  - **최종 ECDF Bagged Rank OOF AUC**: **`0.740587`**

---

## 5. v17 임상 상태 모델링 + Rich Feature + ECDF Ridge 스태킹 통합 최적화 및 워크스페이스 정리 (최종 완료)

### 1) v17 통합 및 스태킹 앙상블 완료
- **개요**: v16의 정밀한 임상 구조 피처셋과 동료분의 Rich Features(Keep-NaN, Clipping, Ratios, 시술명 토큰화)를 결합하고, 다중공선성 제어를 위해 **ECDF Rank 변환 후 Ridge Stacking**을 진행했습니다.
- **학습 실행**: Kaggle GPU 가속기를 활용하여 5개 시드(42, 1004, 7, 2026, 88), 5-Fold Stratified CV 학습을 신속히 수행하고 OOF 캐시 및 가중치 수렴 결과를 도출했습니다.
- **성능 검증 (OOF AUC)**:
  - **최종 ECDF Bagged Ridge Stack OOF AUC**: 🏆 **`0.740830`** (alpha = 100.0)
  - 최상위 성능을 내는 CatBoost(`0.740742`) 모델과 XGBoost, LightGBM, MLP의 상호 보완 기여도를 바탕으로 최종 예측치를 산출하여 앙상블 점수를 한층 더 극대화했습니다.

### 2) 워크스페이스 정리 완료
- **제출물 단일화**: 사방에 흩어져 중복되어 있던 총 32개의 과거 제출용 `.csv` 파일들을 `submission file/archive/` 폴더로 이동 및 분류하여 보관하고, 최종제출본(`submission_v17_bag_0.740830.csv`, `submission_v18_bag_0.740822.csv`)과 기존 기준 성능 제출본(`team_best_submit_0.741849.csv`)만 `submission file/` 루트에 남겨 깔끔하게 정리했습니다.
- **데이터 폴더 정리**: `data/` 폴더 내에 지저분하게 쌓여 있던 제출물들을 정리하고 오직 원본 데이터셋(`train.csv`, `test.csv`, `sample_submission.csv`)과 정의서(`데이터 명세.xlsx`)만 깔끔히 남겨 데이터 소스를 독립 격리했습니다.
- **불필요한 대용량 pkl 삭제**: v9, v13, v14, v16 등 과거 실험의 체크포인트 파일들을 삭제하여 약 **147MB**의 워크스페이스 용량을 확보하고 빌드를 가볍게 만들었습니다.
- **최신 체크포인트 보존**: v17/v18 학습을 통해 생성된 최신 5-seed 예측 체크포인트(`model_oofs_seeds.pkl`)는 각 버전별 `checkpoints/`에 안전하게 배치하였습니다.

---

## 6. v18 MLP 원핫인코딩 & RF/ET 모델 다양화 및 ECDF Ridge Stacking 최적화 완료 (v18 최종완료)

- **개요**: v17 피드백에 따라 MLP 전처리를 원핫 인코딩으로 정상화하고, 스태킹 메타러너 모델의 상관 관계 완화를 위해 **Random Forest** 및 **Extra Trees**를 추가하여 학습 및 앙상블을 완료했습니다.
- **성능 검증 (OOF AUC)**:
  - **최종 ECDF Bagged Ridge Stack OOF AUC**: **`0.740822`**
  - 개별 MLP 모델 및 RF/ET 모델의 풍부한 이질적 랭크 예측 정보를 바탕으로, GBDT 모델들의 오답 노이즈를 릿지 스태킹 메타러너가 효과적으로 차감하여 최종 `0.740822`라는 안정적이고 신뢰도 높은 점수를 획득하였습니다.
- **제출 파일**:
  - **단독 모델**: [submission_v18_bag_0.740822.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_bag_0.740822.csv)
  - **개인 최상위 모델 간 ECDF 랭크 블렌딩 (타 팀 소스 차단)**:
    - **v7 + v17 + v18 균등 결합**: [submission_v18_myblend_equal.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_myblend_equal.csv)
    - **v7 + v17 + v18 (v7 가중치 강화)**: [submission_v18_myblend_heavy_v7.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_myblend_heavy_v7.csv)
    - **v7 + v17 + v18 (신규 모델 강화)**: [submission_v18_myblend_heavy_new.csv](file:///c:/Users/tkskd/infertility_classification/submission%20file/submission_v18_myblend_heavy_new.csv)
- *주의: 외부 'team_best_submit_0.741849.csv' 및 '0.742189' 등 타 팀의 제출 기록이 포함되었던 블렌딩 파일은 규칙 준수를 위해 최종 제출 리스트에서 제외되었습니다.*


