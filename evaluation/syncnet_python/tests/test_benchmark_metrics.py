import csv

import cv2
import numpy as np

from benchmark_metrics import compute_au_f1, compute_cpbd_frame, frechet_distance


def test_frechet_distance_is_zero_for_identical_features():
    features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    assert abs(frechet_distance(features, features)) < 1e-8


def test_cpbd_decreases_after_blur():
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(
        image, "BENCHMARK", (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
        1.2, 255, 3, cv2.LINE_AA,
    )
    blurred = cv2.GaussianBlur(image, (15, 15), 4.0)
    assert compute_cpbd_frame(image) > compute_cpbd_frame(blurred)


def test_au_f1_micro_average(tmp_path):
    columns = ["frame", "AU01_c", "AU02_c"]
    gt_rows = [[1, 1, 0], [2, 0, 1]]
    pred_rows = [[1, 1, 0], [2, 1, 1]]
    paths = []
    for name, rows in (("gt.csv", gt_rows), ("pred.csv", pred_rows)):
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        paths.append(path)

    result = compute_au_f1(*paths)
    assert result["au_precision"] == 2 / 3
    assert result["au_recall"] == 1.0
    assert result["au_f1"] == 0.8
