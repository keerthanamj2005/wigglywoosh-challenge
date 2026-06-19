#Multimodal Dog Activity Detection

A CPU-efficient activity recognition pipeline that combines video motion analysis and IMU sensor data to classify a dog's activity as **Active** or **Static**.

Developed as part of the **WigglyWoosh Technical Challenge**.

---

#Overview

This project uses synchronized video and collar IMU data to determine the dog's activity state.

The solution combines:

* Video-based motion estimation
* Accelerometer and gyroscope measurements
* Sensor fusion logic
* Temporal smoothing

The output is generated as a `timeline.json` file sampled at **2 Hz**.

---
#Features

* CPU-only execution
* Lightweight implementation
* Dense Optical Flow based motion estimation
* IMU feature extraction using accelerometer and gyroscope data
* Sensor fusion for robust predictions
* Temporal smoothing to reduce flickering
* Generates activity timeline with confidence scores

---

#Repository Structure

```text
.
├── run_pipeline.py
├── solution.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

#Installation

Clone the repository:

```bash
git clone https://github.com/keerthanamj2005/wigglywoosh-challenge.git

cd wigglywoosh-challenge
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#Usage

Run:

```bash
python run_pipeline.py Dog_Video.mp4 collar_imu.csv
```

The script generates:

```text
timeline.json
```

---

# Output Format

The output file contains entries sampled every 500 ms.

Example:

```json
{
    "timestamp_ms": 5000,
    "activity": "Active",
    "confidence": 0.727
}
```

---

#Methodology

### Video Analysis

* OpenCV Dense Optical Flow (Farneback algorithm)
* Motion magnitude estimation
* Active / Static classification

### IMU Processing

Acceleration magnitude:

```python
acc_mag = sqrt(ax² + ay² + az²)
```

Gyroscope magnitude:

```python
gyro_mag = sqrt(gx² + gy² + gz²)
```

Combined IMU score:

```python
imu_score = 0.7 * acc_mag + 0.3 * gyro_mag
```

---

#Sensor Fusion

The final activity state is obtained by combining video and IMU predictions.

* Strong IMU confidence overrides video predictions.
* Strong video confidence overrides IMU predictions.
* Weighted voting combines both modalities.
* Hysteresis prevents rapid state changes.

This makes the system robust to noisy video and temporary visual obstructions.

---

## 📦 Output

The generated `timeline.json` contains:

* `timestamp_ms`
* `activity`
* `confidence`

---

##Technologies Used

* Python
* OpenCV
* NumPy
* Pandas
* TensorFlow

---

#Designed For

**WigglyWoosh Technical Challenge**
