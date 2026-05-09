# 🏛️ AI-Powered Virtual Renovation: The Architectural Visionary Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Midjourney](https://img.shields.io/badge/Midjourney-Generative%20AI-purple.svg)](https://www.midjourney.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Object%20Detection-orange.svg)](https://roboflow.com/)

> **"Transforming static architectural photography into dynamic design canvases through the fusion of Generative AI and Geometric Computer Vision."**

---

## 🌟 Vision & Executive Summary

In the modern PropTech and Real Estate landscape, visual storytelling is the difference between a listing and a sale. This repository implements a **High-Fidelity Virtual Renovation Pipeline** that automates the complex process of architectural element replacement.

By bridging the gap between **Generative Artificial Intelligence (Midjourney)** and **Spatial Computer Vision (OpenCV)**, we allow architects, real estate agents, and interior designers to visualize structural changes—starting with windows—with pixel-perfect perspective accuracy.

---

## 🎯 Core Value Propositions

### 💼 For Business Leaders (ROI & Strategy)
*   **Cost Reduction**: Eliminate the need for expensive 3D renders for simple visual upgrades.
*   **Time-to-Market**: Generate realistic renovation previews in seconds, not days.
*   **Client Engagement**: Provide interactive "What-If" scenarios during the sales process.

### 🛠️ For Technical Engineers (Architecture & Logic)
*   **Perspective-Aware Replacement**: Unlike simple overlays, this pipeline uses 4-point Homography to ensure the new assets obey the laws of 3D space.
*   **Seamless Integration**: Modular architecture that can swap Midjourney for Stable Diffusion or Roboflow for custom YOLO models.
*   **Bitwise Blending**: Advanced masking techniques that prevent "sticker-on-photo" artifacts.

---

## 🏗️ Technical Deep Dive: The "Zero to Hero" Workflow

The pipeline operates as a four-stage engine, meticulously moving from raw pixel data to an architecturally sound output.

### 1. Intelligent Asset Detection 🔍
*   **Technology**: Roboflow Inference API.
*   **Process**: The system scans the source image (e.g., `home.jpg`) to locate existing architectural features. It extracts precise bounding box coordinates and confidence scores.
*   **Goal**: Define the "Region of Interest" (ROI) where the renovation will occur.

### 2. Generative Design Synthesis 🎨
*   **Technology**: Midjourney API Integration (`midjourney_api.py`).
*   **Process**: A prompt-based engine requests high-end architectural assets. The system handles remote API handshakes, error recovery (for ngrok/proxy environments), and local asset caching.
*   **Output**: A high-resolution, design-consistent asset (e.g., `window.jpg`).

### 3. Spatial Geometric Alignment 📐
*   **Technology**: OpenCV Homography Transformation.
*   **The Logic**:
    - Real-world photos have perspective distortion (vanishing points).
    - We calculate a **3x3 Transformation Matrix** using `cv2.getPerspectiveTransform`.
    - This maps the flat generative asset into the tilted plane of the house wall using `cv2.warpPerspective`.
*   **Result**: The window doesn't just "sit" on the wall; it is visually "embedded" into it.

### 4. Seamless Neural Blending 🧪
*   **Technology**: Bitwise Masking & Alpha Compositing.
*   **Process**:
    - Creation of a **Binary Mask** of the warped asset.
    - **Bitwise-AND** operations to "cut out" the old window area from the original house.
    - **Bitwise-OR** operations to merge the new warped asset into the vacated space.
*   **Result**: A unified `modified_image.jpg` with zero edge bleed.

---

## 🚀 Getting Started

### 📦 Installation
```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install core dependencies
pip install -r requirements.txt
```

### ⚙️ Configuration
Update `midjourney_api.py` with your service URL:
```python
# Change the base_url to your active Midjourney API bridge
base_url = "http://your-service-url.ngrok.io"
```

### 🏃 Execution
```bash
python3 main.py
```

---

## 📊 Performance & Artifacts

| Artifact | Role | Description |
| :--- | :--- | :--- |
| `home.jpg` | **Input** | The original architectural photograph. |
| `window.jpg` | **Asset** | AI-generated window from the Midjourney pipeline. |
| `prediction.jpg` | **Debug** | Visualization of the object detection boundaries. |
| `modified_image.jpg` | **Output** | The final high-fidelity virtual renovation. |

---

## 🛤️ Future Roadmap

- [ ] **Edge AI Support**: Porting the inference engine to run on mobile devices.
- [ ] **3D Depth Estimation**: Using MiDaS or DPT to handle occlusions (e.g., tree branches in front of windows).
- [ ] **Multi-Asset Batching**: Replacing all windows and doors in a single pass.
- [ ] **Dynamic Lighting**: Adjusting the AI asset's brightness/contrast to match the time-of-day in the source photo.

---

## 👥 Contributors & Philosophy

This project is built on the principle of **Open PropTech**. We believe that the future of architecture is collaborative, generative, and highly visual.

**Developed with ❤️ by the AI Engineering Team.**
