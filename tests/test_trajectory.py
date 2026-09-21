from pathlib import Path

import numpy as np

from trajectory import camera_landmarks_to_world_car, read_bboxes, robust_point


def test_coordinate_conversion_straight_motion():
    # Light is fixed at the origin. Car moves from x=-10 m to x=-6 m.
    observed_light = np.array([[10.0, 0.0, 4.0], [8.0, 0.0, 4.0], [6.0, 0.0, 4.0]])
    actual = camera_landmarks_to_world_car(observed_light)
    expected = np.array([[-10.0, 0.0], [-8.0, 0.0], [-6.0, 0.0]])
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_initial_line_is_aligned_with_world_x():
    observed_light = np.array([[10.0, 2.0, 4.0], [9.0, 1.5, 4.0]])
    actual = camera_landmarks_to_world_car(observed_light)
    assert abs(actual[0, 1]) < 1e-8
    assert actual[0, 0] < 0


def test_robust_patch_ignores_invalid_and_outlier_values():
    points = np.full((9, 9, 3), [10.0, 1.0, 4.0], dtype=float)
    points[4, 4] = [np.nan, np.nan, np.nan]
    points[3, 3] = [1000.0, 1000.0, 1000.0]
    estimate = robust_point(points, 4, 4, radius=2)
    np.testing.assert_allclose(estimate, [10.0, 1.0, 4.0])


def test_actual_challenge_csv_column_names(tmp_path):
    path = tmp_path / "bbox_light.csv"
    path.write_text("frame,x1,y1,x2,y2\n0,378,446,398,494\n1,0,0,0,0\n")
    boxes = read_bboxes(path)
    assert boxes[0].frame_id == "0"
    assert boxes[0].center == (388.0, 470.0)
    assert boxes[0].is_valid
    assert not boxes[1].is_valid
