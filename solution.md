WigglyWoosh Technical Challenge
Overview

This solution detects whether a dog is Active or Static by combining information from both video frames and collar IMU sensor data. The pipeline is designed to run efficiently on a standard laptop CPU and produces a timeline.json file sampled at 2 Hz.

Video Classification

For video analysis, OpenCV Dense Optical Flow (Farneback algorithm) is used to estimate motion between consecutive frames.

Procedure
Frames are sampled every 500 ms.
Consecutive frames are converted to grayscale.
Dense optical flow is computed.
The average motion magnitude is used as a motion score.
Motion score > 1.0 is classified as Active, otherwise Static.
Confidence is computed as:
video_confidence = min(motion_score / 3, 1)

This approach is lightweight and suitable for CPU-only execution.

IMU Feature Extraction

The collar sensor contains accelerometer and gyroscope measurements.

Acceleration Magnitude
acc_mag = √(ax² + ay² + az²)
Gyroscope Magnitude
gyro_mag = √(gx² + gy² + gz²)
Combined IMU Score
imu_score = 0.7 × acc_mag + 0.3 × gyro_mag

The score is normalized to obtain an IMU confidence value between 0 and 1.

Sensor Fusion

The final activity state is obtained by combining video and IMU predictions.

Fusion Rules
Strong IMU confidence overrides video predictions.
Strong video confidence overrides IMU predictions.
Otherwise, weighted voting is used:
score = 0.6 × video_confidence + 0.4 × imu_confidence
If the score exceeds the threshold, the state is classified as Active, otherwise Static.
Temporal Smoothing

Hysteresis is applied to reduce rapid fluctuations between states and improve stability.

Output

The pipeline generates a file named timeline.json sampled at 2 Hz (every 500 ms).

Each entry contains:

timestamp_ms
activity
confidence

Example:

{
    "timestamp_ms": 5000,
    "activity": "Active",
    "confidence": 0.727
}
Design Goals
CPU-only execution
Lightweight implementation
Robustness to noisy video conditions
Sensor fusion between vision and IMU signals
Stable activity predictions