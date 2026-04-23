from pathlib import Path

from nutrilens_ml.data.qa import check_plate_sample, check_pour_sample, run_qa
from nutrilens_ml.data.schemas import (
    FoodLabel,
    LiquidType,
    PlateItem,
    PlateSample,
    PourSample,
)


def _make_plate(tmp_path: Path, *, with_image: bool = True) -> PlateSample:
    image_path = Path("images/a.jpg")
    if with_image:
        (tmp_path / image_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / image_path).write_bytes(b"fake")
    return PlateSample(
        sample_id="p1",
        image_path=image_path,
        items=[PlateItem(label=FoodLabel(name="apple"), grams=120.0)],
    )


def _make_pour(tmp_path: Path, *, total_ml: float = 30.0) -> PourSample:
    video_path = Path("videos/a.mp4")
    (tmp_path / video_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / video_path).write_bytes(b"fake")
    return PourSample(
        sample_id="pour1",
        video_path=video_path,
        liquid=LiquidType.olive_oil,
        total_ml=total_ml,
    )


def test_plate_ok(tmp_path: Path) -> None:
    sample = _make_plate(tmp_path)
    assert check_plate_sample(sample, dataset_root=tmp_path).ok


def test_plate_missing_image(tmp_path: Path) -> None:
    sample = _make_plate(tmp_path, with_image=False)
    outcome = check_plate_sample(sample, dataset_root=tmp_path)
    assert not outcome.ok and "image missing" in (outcome.reason or "")


def test_pour_ok(tmp_path: Path) -> None:
    sample = _make_pour(tmp_path)
    assert check_pour_sample(sample, dataset_root=tmp_path).ok


def test_pour_non_monotonic(tmp_path: Path) -> None:
    sample = _make_pour(tmp_path)
    sample = sample.model_copy(update={"per_frame_cumulative_ml": [0.0, 5.0, 3.0, 8.0]})
    outcome = check_pour_sample(sample, dataset_root=tmp_path)
    assert not outcome.ok and "monotonic" in (outcome.reason or "")


def test_run_qa_quarantines(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()
    quarantine = tmp_path / "q"
    bad = _make_plate(dataset_root, with_image=False)
    outcomes = run_qa([bad], [], dataset_root=dataset_root, quarantine_root=quarantine)
    assert not outcomes[0].ok
    assert (quarantine / "p1" / "reason.txt").is_file()
