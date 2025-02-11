import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import csv
import datetime
import atexit
from typing import Union, List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition")
    parser.add_argument(
        "--det-weight",
        type=str,
        default=r"D:\BTL\face-reidentification\weights\det_10g.onnx",
        help="Path to detection model"
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default=r"D:\BTL\face-reidentification\weights\w600k_r50.onnx",
        help="Path to recognition model"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces"
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default=r"D:\BTL\face-reidentification\faces",
        help="Path to faces stored dir"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"D:\BTL\face-reidentification\assets\duck.mp4",
        help="Video file or video camera source. i.e 0 - webcam"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=0,
        help="Maximum number of face detections from a frame"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="attendance.csv",
        help="Path to save attendance CSV file"
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def build_targets(detector, recognizer, params: argparse.Namespace) -> List[Tuple[np.ndarray, str]]:
    targets = []
    for filename in os.listdir(params.faces_dir):
        name = filename[:-4]
        image_path = os.path.join(params.faces_dir, filename)
        image = cv2.imread(image_path)
        bboxes, kpss = detector.detect(image, max_num=1)
        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue
        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
    return targets

def save_attendance(csv_path: str, present_names: set) -> None:
    """Lưu danh sách người có mặt vào file CSV."""
    if not present_names:  # Không lưu nếu không có ai được nhận diện
        return

    file_exists = os.path.isfile(csv_path)  # Kiểm tra file đã tồn tại chưa
    with open(csv_path, mode='a', newline='') as file:  # Mở file ở chế độ append
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Status", "Timestamp"])  # Thêm header nếu file mới

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name in present_names:
            writer.writerow([name, "Present", now])

    print(f"✅ Attendance saved successfully to {csv_path}")  # In thông báo khi lưu thành công


def frame_processor(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    targets: List[Tuple[np.ndarray, str]],
    colors: dict,
    params: argparse.Namespace,
    present_names: set
) -> np.ndarray:
    """
    Process a video frame for face detection and recognition.

    Args:
        frame (np.ndarray): The video frame.
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.
        present_names (set): Set to store recognized names.

    Returns:
        np.ndarray: The processed video frame.
    """
    bboxes, kpss = detector.detect(frame, params.max_num)

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
            present_names.add(best_match_name)  
        else:
            draw_bbox(frame, bbox, (255, 0, 0))

    return frame


def main(params):
    setup_logging(params.log_level)

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    present_names = set()  # Tạo tập hợp để lưu người có mặt

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_processor(frame, detector, recognizer, targets, colors, params, present_names)
        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    save_attendance(params.csv_path, present_names)  # Lưu dữ liệu khi kết thúc


if __name__ == "__main__":
    args = parse_args()
    if args.source.isdigit():
        args.source = int(args.source)

    present_names = set()  # Biến lưu danh sách người có mặt
    atexit.register(lambda: save_attendance(args.csv_path, present_names))  # Chỉ lưu khi có người
    main(args)
