# RBC2026: Modular Vision System for Robocon

A high-performance, professional-grade vision pipeline designed for the **Robocon 2026** competition. This system integrates YOLOv8 target detection with specialized CNN classification, Kalman filtering for smooth tracking, and a modular architecture optimized for deployment on portable devices like **Surface Go 2**.

---

## 🚀 Key Features

-   **Dual-Model Orchestration**: Seamlessly toggle between `SpearHead` (YOLO detection) and `KFS` (YOLO + CNN verification) in real-round scenarios.
-   **OpenVINO Acceleration**: Full optimization for Intel hardware (CPU/iGPU) using OpenVINO for sub-30ms inference.
-   **Professional Architecture**: Clean separation of concerns with a shared `core/` package and lightweight variant launchers.
-   **Robust Tracking**: Advanced target tracking using **Kalman Filters** and **EMA (Exponential Moving Average)** box smoothing to eliminate jitter.
-   **Portability**: Dynamic project-root detection ensures the system runs anywhere without path modification.
-   **Async Inference**: Multi-threaded processing to maintain high control loop frequency regardless of AI latency.

---

## ✅ Completed Tasks

### [x] Tracking & Label Stabilization
- [x] Implement label hysteresis (persist label during SEARCHING)
- [x] Unify SEARCHING/LOCKED visualization colors (Cyan -> Green)
- [x] Refine `err_x` sending during search transitions

---

## 📁 Project Structure

```text
RBC2026/
├── core/                       # Core "Brain" of the system
│   ├── base_system.py          # Common RoboconSystem logic
│   ├── vision.py               # Integrated Vision Engine (YOLO/CNN)
│   ├── config_manager.py       # Relative path & YAML handler
│   ├── camera.py               # Optimized camera stream
│   ├── label_smoother.py       # Detection stabilization
│   └── utils.py                # Preprocessing & geometry helpers
├── SpearHead_Standard/         # Variant: YOLO only, synchronous
├── SpearHead_HighPerformance/  # Variant: YOLO only, asynchronous (Multi-thread)
├── KFS_Standard/               # Variant: YOLO+CNN, synchronous
├── KFS_HighPerformance/        # Variant: YOLO+CNN, asynchronous (Pro)
├── global_config.yaml          # Universal system configuration
└── system_manager.py           # Master orchestrator for competition
```

---

## 🔧 Installation

### 1. Requirements
- Linux (Ubuntu recommended)
- Python 3.10+
- OpenVINO Toolkit
- OpenCV, Numpy, PyYAML, Serial

### 2. Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd RBC2026

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### 1. Integrated System (Recommended)
The **SystemManager** is designed for the actual match, allowing real-time state switching and global control.
```bash
python3 system_manager.py
```
-   **Key 1**: Switch to SpearHead Mode (Fast Search)
-   **Key 2**: Switch to KFS Mode (Verification & Lock)
-   **Key 0**: Idle Mode
-   **Key Q**: Quit

### 2. Standalone Variants
Each folder contains a `main.py` for testing specific configurations:
```bash
python3 SpearHead_HighPerformance/main.py
```
- **Giao tiếp UART 2 chiều**: Hệ thống nhận lệnh từ Robot qua các ký tự `'0'`, `'1'`, `'2'` để chuyển trạng thái tương ứng.
- **Ổn định mục tiêu (Hysteresis)**: Giữ nhãn và màu xanh lá ổn định ngay cả khi AI bị mất khung hình nhẹ (Kalman prediction), giúp loại bỏ hiện tượng nháy màu xanh dương (Cyan).
- **Tên gọi chuyên nghiệp**: Sử dụng `system_manager.py` (Lớp `SystemManager`) và `global_config.yaml` theo tiêu chuẩn công nghiệp.

### 3. Test Mode (Debugging)
Enable `test_image: true` in `global_config.yaml` to run inference on static images with all bounding boxes rendered (Green for REAL, Gray for FAKE).

---

## ⚙️ Configuration (`global_config.yaml`)

Edit this file to adjust thresholds and hardware settings:
-   `initial_state`: Set the starting mode (1 for Target, 2 for KFS).
-   `device`: Set to `GPU` for Surface Go 2 iGPU or `CPU` for standard laptop usage.
-   `target_types`: Define CNN labels considered as valid targets (e.g., `["REAL", "R1"]`).

---

## 🛡 License & Credits
Developed by **Du** for the **UTC-DKH Team**. Built with ❤️ using **OpenVINO** and **YOLOv8**.
