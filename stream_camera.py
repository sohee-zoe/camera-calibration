import socket
import numpy as np
import cv2
import argparse
from pathlib import Path
from datetime import datetime
from camcalib_yaml import load_calibration_from_yaml


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def undistort_frame(frame, K, D):
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    new_D = np.zeros_like(D)
    # alpha:
    # 0에 가까울수록 왜곡을 펼 때 잘라낸 부분들을 더 보여줌
    # 전체를 보고 싶다면 1, 펴진 부분만 보고 싶다면 0에 가깝게
    frame = cv2.undistort(frame, K, D, None, new_K)
    return frame, new_K, new_D

def corner_points(corner):
    corner_points = corner[0]
    center = np.mean(corner_points, axis=0).astype(int)

    corner = np.array(corner).reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corner
    return center, topLeft, topRight, bottomRight, bottomLeft

def corner_to_points(topLeft, topRight, bottomRight, bottomLeft):
    topRightPoint = (int(topRight[0]), int(topRight[1]))
    topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
    bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))
    return topRightPoint, topLeftPoint, bottomRightPoint, bottomLeftPoint

def to_pos(tvec):
    x = round(tvec[0][0], 2)
    y = round(tvec[1][0], 2)
    z = round(tvec[2][0], 2)
    return x, y, z

def to_rot(rvec):
    rx = round(np.rad2deg(rvec[0][0]), 2)
    ry = round(np.rad2deg(rvec[1][0]), 2)
    rz = round(np.rad2deg(rvec[2][0]), 2)
    return rx, ry, rz

def draw_custom_axes(image, K, D, rvec, tvec, length=0.05):
    # 좌표계 기준점
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    x_axis = np.array([[length, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, length, 0]], dtype=np.float32)
    z_axis = np.array([[0, 0, length]], dtype=np.float32)

    # 3D → 2D 변환
    imgpts_origin, _ = cv2.projectPoints(origin, rvec, tvec, K, D)
    imgpts_x, _ = cv2.projectPoints(x_axis, rvec, tvec, K, D)
    imgpts_y, _ = cv2.projectPoints(y_axis, rvec, tvec, K, D)
    imgpts_z, _ = cv2.projectPoints(z_axis, rvec, tvec, K, D)

    p_origin = tuple(imgpts_origin[0].ravel().astype(int))
    p_x = tuple(imgpts_x[0].ravel().astype(int))
    p_y = tuple(imgpts_y[0].ravel().astype(int))
    p_z = tuple(imgpts_z[0].ravel().astype(int))

    # 선 그리기 (BGR 순서)
    cv2.line(image, p_origin, p_x, (0, 0, 255), 3)  # X - 빨강
    cv2.line(image, p_origin, p_y, (0, 255, 0), 3)  # Y - 초록
    cv2.line(image, p_origin, p_z, (255, 0, 0), 3)  # Z - 파랑

    return image


def detect_aruco(frame, K=None, D=None, aruco_type="DICT_6X6_250", aruco_length=0.035):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if K is not None and D is not None:
            for i, corner in enumerate(corners):
                # 마커의 3D 좌표 (시계방향, 기준: 중심 (0,0,0))
                obj_points = np.array([
                    [-aruco_length / 2, aruco_length / 2, 0], # 좌상
                    [aruco_length / 2, aruco_length / 2, 0], # 우상
                    [aruco_length / 2, -aruco_length / 2, 0], # 우하
                    [-aruco_length / 2, -aruco_length / 2, 0]  # 좌하
                ], dtype=np.float32)

                img_points = corner[0].astype(np.float32)

                center, topLeft, topRight, bottomRight, bottomLeft = corner_points(corner)
                # topRightPoint, topLeftPoint, bottomRightPoint, bottomLeftPoint = corner_to_points(topLeft, topRight, bottomRight, bottomLeft)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE,)
                if success:
                    # 좌표축 그리기
                    cv2.drawFrameAxes(frame, K, D, rvec, tvec, aruco_length * 0.5)
                    # frame = draw_custom_axes(frame, K, D, rvec, tvec, length=aruco_length * 0.5)

                    r = rvec.flatten()
                    t = tvec.flatten()
                    distance = np.linalg.norm(t)
                    print(f"[INFO] ID {ids[i][0]} | rvec: {r} | tvec: {t} | 거리: {distance:.2f}m")

                    x, y, z = to_pos(tvec)
                    rx, ry, rz = to_rot(rvec)

                    id_text = f"ID = {ids[i][0]}"
                    pos_text = f"Pos: ({x:.2f}, {y:.2f}, {z:.2f}) m"
                    rot_text = f"Rot: ({rx:.1f}, {ry:.1f}, {rz:.1f}) deg"

                    cv2.putText(frame,
                                id_text,
                                (center[0] + 10, center[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

                    cv2.putText(frame,
                                pos_text,
                                (int(topLeft[0] - 10), center[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)

                    cv2.putText(frame,
                                rot_text,
                                (int(topLeft[0] - 10), center[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)
    return frame


def process_frame(frame, window_title="Camera Feed", K=None, D=None,
                  aruco_type="DICT_6X6_250", aruco_length=0.035):
    if frame is None or frame.size == 0:
        print("[WARN] 비어있는 프레임 - 표시 생략")
        return None

    if K is not None and D is not None:
        frame, new_K, new_D = undistort_frame(frame, K, D)
        frame = detect_aruco(frame, K=new_K, D=new_D, aruco_type=aruco_type, aruco_length=aruco_length)
    else:
        frame = detect_aruco(frame, K=K, D=D, aruco_type=aruco_type, aruco_length=aruco_length)

    cv2.imshow(window_title, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return "quit"
    elif key == ord("s"):
        filename = datetime.now().strftime("frames/%Y%m%d_%H%M%S.jpg")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename, frame)
        print(f"[INFO] 이미지 저장됨: {filename}")
    return None


def receive_udp_stream(udp_ip, udp_port, buffer_size, K=None, D=None,
                       aruco_type="DICT_6X6_250", aruco_length=0.035):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(1.0)
    print(f"[INFO] UDP 수신 대기 중: {udp_ip}:{udp_port}")

    data_buffer = bytearray()
    expected_size = 0
    receiving_data = False
    last_frame = None

    try:
        while True:
            try:
                data, addr = sock.recvfrom(buffer_size)

                if len(data) == 4 and not receiving_data:
                    expected_size = int.from_bytes(data, byteorder="big")
                    data_buffer = bytearray()
                    receiving_data = True
                    continue

                if receiving_data:
                    if data == b"END":
                        if len(data_buffer) >= expected_size:
                            img_data = np.frombuffer(
                                data_buffer[:expected_size], dtype=np.uint8
                            )
                            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                            if frame is not None:
                                last_frame = frame.copy()
                                result = process_frame(last_frame, "Camera Feed (UDP)", K, D,
                                                       aruco_type, aruco_length)
                                if result == "quit":
                                    break
                        else:
                            print(f"[WARN] 프레임 크기 부족: {len(data_buffer)} / {expected_size}")

                        receiving_data = False
                        data_buffer = bytearray()
                    else:
                        data_buffer.extend(data)

            except socket.timeout:
                print("타임아웃: 데이터 수신 대기 중...")
                receiving_data = False
    finally:
        sock.close()
        cv2.destroyAllWindows()


def receive_usb_camera(camera_index=0, K=None, D=None,
                       aruco_type="DICT_6X6_250", aruco_length=0.035):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] USB 카메라를 열 수 없습니다.")
        return

    # 자동 초점 끄기 시도
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = 자동 초점 끄기 (1 = 켜기)
    # cap.set(cv2.CAP_PROP_FOCUS, 30)  # 수동 포커스 (0~255)

    print("[INFO] USB 카메라 스트리밍 시작")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 프레임 수신 실패")
                continue

            result = process_frame(frame, "Camera Feed (USB)", K, D,
                                   aruco_type, aruco_length)
            if result == "quit":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영상 입력 선택")
    parser.add_argument("--source", type=str, choices=["udp", "usb"], default="usb", help="영상 입력 소스 (udp 또는 usb)")
    parser.add_argument("--udp_ip", type=str, default="0.0.0.0", help="UDP 수신 IP")
    parser.add_argument("--udp_port", type=int, default=5000, help="UDP 수신 포트")
    parser.add_argument("--camera_index", type=int, default=0, help="USB 카메라 인덱스")
    parser.add_argument("--calibration", action=argparse.BooleanOptionalAction, default=False, help="캘리브레이션 실행 여부")
    parser.add_argument("--parameter", type=str, default="camera_param.yaml", help="YAML 포맷 카메라 캘리브레이션 파일")
    parser.add_argument("--aruco_type", type=str, default="DICT_6X6_250")
    parser.add_argument("--aruco_length", type=float, default=0.035, help="ArUco Maker 한 칸 크기 (meter)")
    args = parser.parse_args()

    K, D = (None, None)
    if args.calibration and Path(args.parameter).is_file():
            K, D = load_calibration_from_yaml(args.parameter)

    if args.source == "udp":
        receive_udp_stream(args.udp_ip, args.udp_port, 65536, K, D, args.aruco_type, args.aruco_length)
    else:
        receive_usb_camera(args.camera_index, K, D, args.aruco_type, args.aruco_length)
