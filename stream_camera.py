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


def detect_aruco(frame, K=None, D=None, marker_length=0.035, aruco_type = "DICT_6X6_250"):
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
                    [-marker_length / 2,  marker_length / 2, 0], # 좌상
                    [ marker_length / 2,  marker_length / 2, 0], # 우상
                    [ marker_length / 2, -marker_length / 2, 0], # 우하
                    [-marker_length / 2, -marker_length / 2, 0]  # 좌하
                ], dtype=np.float32)

                img_points = corner[0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE,)
                if success:
                    # 좌표축 그리기
                    cv2.drawFrameAxes(frame, K, D, rvec, tvec, marker_length * 0.5)

                    r = rvec.flatten()
                    t = tvec.flatten()
                    distance = np.linalg.norm(t)
                    print(f"[INFO] ID {ids[i][0]} | rvec: {r} | tvec: {t} | 거리: {distance:.2f}m")

                    corner_points = corner[0]
                    center = np.mean(corner_points, axis=0).astype(int)
                    text = f"ID:{ids[i][0]} | {t[0]:.2f},{t[1]:.2f},{t[2]:.2f} m | Dist:{distance:.2f}m"
                    cv2.putText(frame, text, (center[0] - 50, center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    return frame



def process_frame(frame, window_title="Camera Feed", K=None, D=None):
    if frame is None or frame.size == 0:
        print("[WARN] 비어있는 프레임 - 표시 생략")
        return None

    if K is not None and D is not None:
        frame, new_K, new_D = undistort_frame(frame, K, D)
        frame = detect_aruco(frame, K=new_K, D=new_D)
    else:
        frame = detect_aruco(frame, K=K, D=D)

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


def receive_udp_stream(udp_ip, udp_port, buffer_size, K=None, D=None):
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
                                result = process_frame(last_frame, "Camera Feed (UDP)", K, D)
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


def receive_usb_camera(camera_index=0, K=None, D=None):
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

            result = process_frame(frame, "Camera Feed (USB)", K, D)
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
    parser.add_argument("--calib", type=str, default="camera_param.yaml", help="YAML 포맷 카메라 캘리브레이션 파일")
    args = parser.parse_args()

    K, D = (None, None)
    if args.calib:
        if Path(args.calib).is_file():
            K, D = load_calibration_from_yaml(args.calib)

    if args.source == "udp":
        receive_udp_stream(args.udp_ip, args.udp_port, 65536, K, D)
    else:
        receive_usb_camera(args.camera_index, K, D)
