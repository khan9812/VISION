"""Review the 21 zero-classifiable Shape images without rerunning models.

The script copies the frozen SAM2/Hungarian annotations to a separate working
directory, identifies images with no non-skipped GT label in the frozen source,
and opens the saved-mask review GUI only for those images.  Every displayed mask
must receive either a shape label or an explicit non-particle decision.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from shape_validation import ShapeValidator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "workspace_outputs" / "validation"


def annotation_stem(path: Path) -> str:
    suffix = "_shape.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected annotation filename: {path}")
    return path.name[: -len(suffix)]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def zero_classifiable_stems(annotations_dir: Path) -> list[str]:
    stems = []
    for path in sorted(annotations_dir.glob("*_shape.json")):
        payload = load_json(path)
        valid_rows = [
            row
            for row in payload.get("particles", [])
            if not bool(row.get("skipped", False)) and row.get("gt_shape")
        ]
        if not valid_rows:
            stems.append(annotation_stem(path))
    return stems


def prepare_review_annotations(source_dir: Path, review_dir: Path) -> None:
    source_files = sorted(source_dir.glob("*_shape.json"))
    if not source_files:
        raise FileNotFoundError(f"No source annotations found in {source_dir}")
    review_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_files:
        destination = review_dir / source_path.name
        if not destination.exists():
            shutil.copy2(source_path, destination)


def verify_saved_masks(stems: list[str], predictions_dir: Path) -> tuple[int, list[str]]:
    candidate_count = 0
    problems = []
    for stem in stems:
        prediction_path = predictions_dir / f"{stem}_pred.json"
        if not prediction_path.exists():
            problems.append(f"{stem}: missing {prediction_path.name}")
            continue
        prediction = load_json(prediction_path)
        mask_ids = {
            int(row["particle_id"])
            for row in prediction.get("particles", [])
            if row.get("mask") is not None and row.get("mask_shape") is not None
        }
        candidate_count += len(mask_ids)
        if not mask_ids:
            problems.append(f"{stem}: prediction JSON has no saved masks")
    return candidate_count, problems


def review_completion(
    stems: list[str], source_dir: Path, review_dir: Path
) -> tuple[list[str], dict[str, list[int]]]:
    incomplete = []
    missing_by_image = {}
    for stem in stems:
        source = load_json(source_dir / f"{stem}_shape.json")
        reviewed = load_json(review_dir / f"{stem}_shape.json")
        source_ids = {int(row["particle_id"]) for row in source.get("particles", [])}
        decided_ids = {
            int(row["particle_id"])
            for row in reviewed.get("particles", [])
            if not bool(row.get("skipped", False)) and row.get("gt_shape")
        }
        metadata = reviewed.get("annotation_metadata", {})
        for entry in metadata.get("skip_review_history", []):
            for key in (
                "reviewed_particle_ids",
                "annotated_particle_ids",
                "kept_skipped_particle_ids",
            ):
                decided_ids.update(int(value) for value in entry.get(key, []))
        missing = sorted(source_ids - decided_ids)
        if missing:
            incomplete.append(stem)
            missing_by_image[stem] = missing
    return incomplete, missing_by_image


def write_completion_manifest(
    stems: list[str], source_dir: Path, review_dir: Path, audit_dir: Path
) -> tuple[dict, Path]:
    incomplete, missing_by_image = review_completion(stems, source_dir, review_dir)
    completion = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_image_count": len(stems),
        "completed_image_count": len(stems) - len(incomplete),
        "incomplete_image_count": len(incomplete),
        "incomplete_images": incomplete,
        "undecided_particle_ids": missing_by_image,
    }
    completion_path = audit_dir / "zero_classifiable_completion.json"
    with completion_path.open("w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, ensure_ascii=False)
    return completion, completion_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review every skipped mask in the frozen zero-classifiable Shape images."
    )
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "Dataset_shape")
    parser.add_argument(
        "--source-annotations-name",
        default="annotations_sam_p095_s080_hungarian",
    )
    parser.add_argument(
        "--review-annotations-name",
        default="annotations_sam_p095_s080_manual_review",
    )
    parser.add_argument("--predictions-name", default="predictions_sam_p095_s080")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-only", action="store_true")
    mode.add_argument("--status-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--evaluation-output-name",
        default="results_publication_final_manual_review",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validation_root = args.validation_root.resolve()
    shape_root = validation_root / "shape_validation"
    source_dir = shape_root / args.source_annotations_name
    review_dir = shape_root / args.review_annotations_name
    predictions_dir = shape_root / args.predictions_name
    review_audit_dir = shape_root / "zero_classifiable_review"
    review_audit_dir.mkdir(parents=True, exist_ok=True)

    targets = zero_classifiable_stems(source_dir)
    candidate_count, mask_problems = verify_saved_masks(targets, predictions_dir)
    if mask_problems:
        raise RuntimeError("Saved-mask preflight failed:\n" + "\n".join(mask_problems))

    target_manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_annotations_dir": str(source_dir.resolve()),
        "review_annotations_dir": str(review_dir.resolve()),
        "predictions_dir": str(predictions_dir.resolve()),
        "zero_classifiable_image_count": len(targets),
        "saved_mask_candidate_count": candidate_count,
        "image_stems": targets,
    }
    manifest_path = review_audit_dir / "zero_classifiable_targets.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(target_manifest, handle, indent=2, ensure_ascii=False)

    print(f"Zero-classifiable images: {len(targets)}")
    print(f"Saved masks to review: {candidate_count}")
    print("Image stems: " + ", ".join(targets))
    print(f"Target manifest: {manifest_path}")
    if args.list_only:
        return

    prepare_review_annotations(source_dir, review_dir)
    if args.status_only:
        completion, completion_path = write_completion_manifest(
            targets, source_dir, review_dir, review_audit_dir
        )
        print(
            f"Review completion: {completion['completed_image_count']}/{len(targets)} images"
        )
        print(f"Completion manifest: {completion_path}")
        return

    validator = ShapeValidator(
        dataset_dir=str(args.dataset_dir.resolve()),
        output_dir=str(validation_root),
        shape_preset="2D",
        pred_iou_thresh=0.95,
        stability_score_thresh=0.80,
        expected_images=100,
    )
    validator.annotations_dir = review_dir
    validator.predictions_dir = predictions_dir
    validator.results_dir = review_audit_dir

    if not args.evaluate_only:
        validator.review_skipped_annotations(
            predictions_dir=predictions_dir,
            image_stems=targets,
            additional_predictions_dirs=[predictions_dir],
            include_previously_reviewed=False,
            require_decision_for_all=True,
            show_predictions_in_gui=False,
        )

    completion, completion_path = write_completion_manifest(
        targets, source_dir, review_dir, review_audit_dir
    )
    incomplete = completion["incomplete_images"]
    print(
        f"Review completion: {completion['completed_image_count']}/{len(targets)} images"
    )
    print(f"Completion manifest: {completion_path}")
    if incomplete:
        print("Review is incomplete; evaluation was not run.")
        print("Run the same command again to resume the remaining images.")
        return

    evaluation_dir = shape_root / args.evaluation_output_name
    evaluation = validator.evaluate_prediction_directory(
        predictions_dir=predictions_dir,
        evaluation_output_dir=evaluation_dir,
        annotation_override_dir=review_dir,
        common_ids_only=False,
        comparison_policy=(
            "Prediction-blinded manual re-review of every saved mask in the 21 originally "
            "zero-classifiable images; strict prediction/annotation ID alignment"
        ),
    )
    print(f"Reviewed evaluation workbook: {evaluation['excel_path']}")
    print(f"Reviewed evaluation manifest: {evaluation['manifest_path']}")


if __name__ == "__main__":
    main()
