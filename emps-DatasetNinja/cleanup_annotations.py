"""
이미지 파일과 매칭되지 않는 JSON 어노테이션 파일 삭제 스크립트
img 폴더에 없는 이미지에 대응하는 ann 폴더의 .json 파일을 제거합니다.
"""
import os
from pathlib import Path


def cleanup_unmatched_annotations(img_dir: str, ann_dir: str, dry_run: bool = True):
    """
    이미지 파일과 매칭되지 않는 어노테이션 파일을 삭제합니다.

    Args:
        img_dir: 이미지 파일이 있는 디렉토리 경로
        ann_dir: 어노테이션 JSON 파일이 있는 디렉토리 경로
        dry_run: True일 경우 실제 삭제하지 않고 결과만 출력 (기본값: True)
    """
    img_path = Path(img_dir)
    ann_path = Path(ann_dir)

    if not img_path.exists():
        print(f"[ERROR] 이미지 디렉토리가 존재하지 않습니다: {img_dir}")
        return

    if not ann_path.exists():
        print(f"[ERROR] 어노테이션 디렉토리가 존재하지 않습니다: {ann_dir}")
        return

    # img 폴더의 모든 이미지 파일명 수집
    image_files = {f.name for f in img_path.iterdir() if f.is_file()}
    print(f"[IMG] 이미지 파일 개수: {len(image_files)}")

    # ann 폴더의 모든 JSON 파일 확인
    json_files = [f for f in ann_path.iterdir() if f.suffix == '.json']
    print(f"[ANN] 어노테이션 파일 개수: {len(json_files)}")

    # 삭제 대상 파일 찾기
    files_to_delete = []
    for json_file in json_files:
        # .json 확장자를 제거한 이미지 파일명
        image_name = json_file.name.replace('.json', '')

        if image_name not in image_files:
            files_to_delete.append(json_file)

    # 결과 출력
    mode_label = "[DRY RUN]" if dry_run else "[DELETING]"
    print(f"\n{mode_label} 삭제 대상 파일: {len(files_to_delete)}개")

    if files_to_delete:
        print("\n삭제될 파일 목록:")
        for i, file in enumerate(files_to_delete, 1):
            print(f"  {i}. {file.name}")
            if not dry_run:
                try:
                    file.unlink()
                    print(f"     [OK] 삭제 완료")
                except Exception as e:
                    print(f"     [FAIL] 삭제 실패: {e}")
    else:
        print("[OK] 삭제할 파일이 없습니다. 모든 어노테이션 파일이 이미지와 매칭됩니다.")

    if dry_run:
        print("\n[INFO] 실제로 삭제하려면 dry_run=False로 실행하세요.")
    else:
        print(f"\n[DONE] 완료: {len(files_to_delete)}개 파일 삭제됨")


if __name__ == "__main__":
    # 경로 설정
    IMG_DIR = "./ds/img"
    ANN_DIR = "./ds/ann"

    # 1단계: dry_run으로 먼저 확인
    print("=" * 60)
    print("1단계: 삭제 대상 파일 확인 (실제 삭제 안 함)")
    print("=" * 60)
    cleanup_unmatched_annotations(IMG_DIR, ANN_DIR, dry_run=True)

    # 2단계: 실제 삭제 (주석 해제하여 실행)
    # print("\n" + "=" * 60)
    # print("2단계: 실제 파일 삭제")
    # print("=" * 60)
    # user_input = input("\n[WARNING] 정말 삭제하시겠습니까? (yes/no): ")
    # if user_input.lower() == 'yes':
    #     cleanup_unmatched_annotations(IMG_DIR, ANN_DIR, dry_run=False)
    # else:
    #     print("취소되었습니다.")
