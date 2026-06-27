"""
최종 제출 재현용 파일들을 한 폴더(final_submission_bundle/)로 모으는 스크립트.

실행 위치: experiment_history/2차/ (다른 노트북들과 같은 위치)
실행 방법: python collect_files_for_bundle.py

이 스크립트는 흩어져 있는 npy 캐시 파일과 팀원 v5 파일을 전부 한 곳에 복사합니다.
복사만 하고 원본은 그대로 두므로, 여러 번 실행해도 안전합니다.
"""
import shutil
from pathlib import Path

# ── 경로 설정 (필요하면 여기만 고치면 됨) ──────────────────────────────
# ★ 실행한 위치(현재 작업 디렉토리)가 아니라, 이 스크립트 파일 자신의 위치를 기준으로 삼음
#   (VS Code의 "실행" 버튼 등이 워크스페이스 루트에서 실행하는 경우에도 항상 정확하게 동작)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../../data"
BLEND_DIR = SCRIPT_DIR / "../blend_cache"
V5_DIR = SCRIPT_DIR / "../팀원파일/김영혜"
BUNDLE_DIR = SCRIPT_DIR / "final_submission_bundle"
# ──────────────────────────────────────────────────────────────────

print(f"스크립트 위치: {SCRIPT_DIR}")

BUNDLE_DIR.mkdir(exist_ok=True)

# 12개 후보 중 이 노트북에서 새로 만드는 3개(donor_cat, alpha_cat, lgbm_spw2_richfeat)는
# 제외하고, "이미 다른 노트북에서 만들어둔" 9개만 모음
REQUIRED_FILES = [
    DATA_DIR / "train.csv",
    DATA_DIR / "test.csv",
    BLEND_DIR / "oof_10seed_bagged.npy",
    BLEND_DIR / "test_lgbm_bagged.npy",
    BLEND_DIR / "oof_xgboost_bagged.npy",
    BLEND_DIR / "test_xgboost_bagged.npy",
    BLEND_DIR / "oof_catboost_bagged.npy",
    BLEND_DIR / "test_catboost_bagged.npy",
    BLEND_DIR / "oof_feature_subspace.npy",
    BLEND_DIR / "test_feature_subspace.npy",
    BLEND_DIR / "xgb_rankpairwise_oof.npy",
    BLEND_DIR / "xgb_rankpairwise_test.npy",
    BLEND_DIR / "lgbm_lambdarank_oof.npy",
    BLEND_DIR / "lgbm_lambdarank_test.npy",
    BLEND_DIR / "oof_lgbm_keepnan.npy",
    BLEND_DIR / "test_lgbm_keepnan.npy",
    BLEND_DIR / "oof_mlp.npy",
    BLEND_DIR / "test_mlp.npy",
    V5_DIR / "v5_ensemble_oof.npy",
    V5_DIR / "submission_v5_imputed.csv",
]

print(f"번들 폴더: {BUNDLE_DIR.resolve()}\n")

missing = []
for src in REQUIRED_FILES:
    if not src.exists():
        # 파일명이 약간 다를 수 있으니(xgb_rankpairwise_test.npy vs test_rankpairwise.npy 등),
        # 못 찾으면 일단 알려주고 계속 진행. 아래 "찾지 못한 파일" 목록을 보고 직접 확인하세요.
        missing.append(src)
        continue
    dst = BUNDLE_DIR / src.name
    shutil.copy2(src, dst)
    print(f"  복사 완료: {src} -> {dst.name}")

if missing:
    print("\n⚠️  찾지 못한 파일 (파일명이 다를 수 있음, blend_cache 폴더를 직접 확인하세요):")
    for m in missing:
        print(f"   - {m}")
else:
    print("\n✅ 전체 파일 복사 완료")

print(f"\n번들 폴더 안 파일 목록:")
for f in sorted(BUNDLE_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<40} {size_kb:>10.1f} KB")
