from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
import json
import sys


# ---------------- VIDEO ANALYZER ----------------
# Video activity estimation using optical flow
class VideoAnalyzer:

    def __init__(self, video_path):
        self.video_path = video_path

        self.model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg'
    )

    def analyze(self):

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        prev_gray = None
        video_results = []

        duration_ms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps*1000)

        duration_ms = min(duration_ms,12000)

        timestamps=np.arange(0,duration_ms,500)

        for t in timestamps:

            frame_no = int((t / 1000) * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:

                prev_gray = gray

                video_results.append({
                    "timestamp_ms": int(t),
                    "video_state": "Static",
                    "video_confidence": 0.5
                })

                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )

            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

            motion_score = mag.mean()

            state = "Active" if motion_score > 1 else "Static"

            confidence = min(float(motion_score / 3), 1)

            video_results.append({
                "timestamp_ms": int(t),
                "video_state": state,
                "video_confidence": confidence
            })

            prev_gray = gray

        cap.release()

        return video_results



# IMU feature extraction using acceleration and gyroscope magnitude
class IMUAnalyzer:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def analyze(self):

        df = pd.read_csv(self.csv_path)

        df["acc_mag"] = np.sqrt(
            df["accel_x"]**2 +
            df["accel_y"]**2 +
            df["accel_z"]**2
        )

        df["gyro_mag"] = np.sqrt(
            df["gyro_x"]**2 +
            df["gyro_y"]**2 +
            df["gyro_z"]**2
        )

        imu_results = []

        duration_ms = int(df["timestamp_ms"].max())

        timestamps=np.arange(0,duration_ms+1,500)

        df["imu_score"] = 0.7 * df["acc_mag"] + 0.3 * df["gyro_mag"]

        min_score = df["imu_score"].min()
        max_score = df["imu_score"].max()

        for t in timestamps:

            window = df[
                (df["timestamp_ms"] >= t) &
                (df["timestamp_ms"] < t + 500)
            ]

            acc_score = window["acc_mag"].mean()
            gyro_score = window["gyro_mag"].mean()

            imu_score = 0.7 * acc_score + 0.3 * gyro_score

            confidence = (imu_score - min_score) / (max_score - min_score)

            confidence = max(0, min(confidence, 1))

            state = "Active" if confidence > 0.5 else "Static"

            imu_results.append({
                "timestamp_ms": int(t),
                "imu_state": state,
                "imu_confidence": float(confidence)
            })

        return imu_results


# ---------------- SENSOR FUSION ----------------
# Sensor fusion logic combining video and IMU predictions
class SensorFusion:

    def fuse(self, video_results, imu_results):

        final_results = []

        prev_state = "Static"

        for video, imu in zip(video_results, imu_results):

            video_conf = float(video["video_confidence"])
            imu_conf = float(imu["imu_confidence"])

            video_state = video["video_state"]
            imu_state = imu["imu_state"]

            # Strong IMU evidence
            if imu_conf > 0.8:

                final_state = imu_state
                final_conf = imu_conf

            # Strong video evidence
            elif video_conf > 0.7:

                final_state = video_state
                final_conf = video_conf

            else:

                score = 0

                if video_state == "Active":
                    score += 0.6 * video_conf

                if imu_state == "Active":
                    score += 0.4 * imu_conf

                if score > 0.5:
                    final_state = "Active"
                else:
                    final_state = "Static"

                final_conf = max(video_conf, imu_conf)

            # Hysteresis: prevent flickering
            if prev_state == "Active" and final_state == "Static":
                if final_conf < 0.3:
                    final_state = "Static"
                else:
                    final_state = "Active"

            prev_state = final_state

            final_results.append({
                "timestamp_ms": int(video["timestamp_ms"]),
                "activity": final_state,
                "confidence": round(float(final_conf), 3)
            })

        return final_results

# ---------------- MAIN ----------------

if __name__ == "__main__":

    if len(sys.argv) == 3:

        video_path = sys.argv[1]
        csv_path = sys.argv[2]

    else:

        video_path = "Dog_Video.mp4"
        csv_path = "collar_imu.csv"

    video = VideoAnalyzer(video_path)
    video_results = video.analyze()
    
    imu = IMUAnalyzer(csv_path)
    imu_results = imu.analyze()

    fusion = SensorFusion()

    final_results = fusion.fuse(
        video_results,
        imu_results
    )
    

    with open("timeline.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("timeline.json generated successfully!")

    
