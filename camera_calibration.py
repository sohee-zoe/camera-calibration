import os
import cv2
import numpy as np
import glob
import argparse
from typing import Tuple, List, Optional
from camcalib_yaml import save_calibration_to_yaml, load_calibration_from_yaml


def setup_criteria() -> Tuple[int, int, float]:
    """코너 서브픽셀 검출 기준 설정"""
    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def setup_object_points(
        num_corners_x: int,
        num_corners_y: int,
        square_size: float
) -> np.ndarray:
    """3D 월드 좌표계 포인트 생성"""
    objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)
    return objp * square_size


def process_images(
        path: str,
        pattern_size: Tuple[int, int],
        criteria: Tuple[int, int, float],
        objp: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    images = glob.glob(os.path.join(path, "*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG images found in {path}")

    object_points = []
    image_points = []
    pattern_detected = 0
    img_shape: Optional[Tuple[int, int]] = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if success:
            if img_shape is None:
                img_shape = gray.shape[::-1]  # (width, height)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners_refined)
            object_points.append(objp.copy())

            # 시각화
            cv2.drawChessboardCorners(img, pattern_size, corners_refined, success)
            cv2.imshow("Detected Pattern", img)
            cv2.waitKey(200)
            pattern_detected += 1

    if pattern_detected == 0 or img_shape is None:
        raise ValueError("Chessboard pattern not detected in any images")

    print(f"Successfully processed {pattern_detected}/{len(images)} images")
    return object_points, image_points, img_shape


def calibrate(
        num_corners_x: int,
        num_corners_y: int,
        square_size: float,
        path: str
) -> Tuple[np.ndarray, np.ndarray]:
    criteria = setup_criteria()
    objp = setup_object_points(num_corners_x, num_corners_y, square_size)
    pattern_size = (num_corners_x, num_corners_y)
    object_points, image_points, img_shape = process_images(
        path, pattern_size, criteria, objp
    )

    # 캘리브레이션 수행 (ret: RMS error)
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, img_shape, None, None
    )

    # 결과 저장 및 검증
    save_calibration_to_yaml("camera_param.yaml", mtx, dist)
    verify_mtx, verify_dist = load_calibration_from_yaml("camera_param.yaml")

    print("\nCalibration Results (Verified)")
    print(f"Camera Matrix:\n{verify_mtx}")
    print(f"\nDistortion Coefficients:\n{verify_dist}")

    cv2.destroyAllWindows()
    return mtx, dist


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("--size",
                        type=lambda s: tuple(map(int, s.split('x'))),
                        default=(8, 6),
                        help="체커보드 코너 개수 (가로x세로), 예: 8x6")
    parser.add_argument("--square",
                        type=float,
                        default=0.025, # 25 mm
                        help="체커보드 한 칸의 실제 크기(미터 단위)")
    parser.add_argument("--path",
                        type=str,
                        default="./frames",
                        help="이미지 저장 경로")

    args = parser.parse_args()

    if len(args.size) != 2:
        raise ValueError("--size argument must be in 'WxH' format")

    calibrate(
        num_corners_x=args.size[0],
        num_corners_y=args.size[1],
        square_size=args.square,
        path=args.path
    )


if __name__ == "__main__":
    main()
