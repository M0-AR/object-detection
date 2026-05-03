# 🏠 AI-Powered Virtual Renovation Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![AI Integration](https://img.shields.io/badge/AI-Roboflow%20%7C%20Midjourney-orange.svg)]()

> **"From Zero to Hero"**: Transform architectural visualization with the power of Generative AI and Computer Vision.

---

## 🌟 Business Overview: The Value Proposition

In the competitive landscapes of **Real Estate**, **Interior Design**, and **Architectural Marketing**, visualization is the primary driver of sales. This repository provides a proof-of-concept for an automated **Virtual Renovation Pipeline**.

### 🚀 Key Business Benefits:
- **Scalable Content Creation:** Automatically replace outdated architectural elements (windows, doors, textures) with modern designs.
- **Cost Efficiency:** Eliminate the need for expensive manual 3D modeling and rendering for simple "what-if" scenarios.
- **Enhanced Personalization:** Allow potential buyers to see property potential instantly by swapping styles (e.g., from Traditional to Modern industrial) using Generative AI.
- **High ROI:** Increase listing engagement by providing high-quality, "staged" visuals that bridge the gap between "what is" and "what could be."

---

## 🛠 Technical Architecture: How it Works

The system operates on a sophisticated 3-stage pipeline that ensures both geometric accuracy and aesthetic quality.

### 1. Object Detection (Spatial Context)
Using **Roboflow's Inference API**, the system identifies specific architectural features (e.g., windows) within a base photograph. It extracts precise bounding boxes and confidence scores, providing the spatial coordinates necessary for transformation.

### 2. Generative Asset Creation (The Design)
The pipeline integrates with **Midjourney's Generative AI** (via a custom API bridge). This allows for the on-demand generation of architectural assets based on natural language prompts (e.g., *"A modern minimalist black-framed window"*), ensuring the design is always cutting-edge.

### 3. Perspective Blending (The "Magic")
Using **OpenCV**, the system performs a **Homography Transformation**. This maps the 2D generated design into the 3D perspective of the original photo.
- **Transformation:** `cv2.getPerspectiveTransform` aligns the asset corners to the detected region.
- **Seamless Blending:** Bitwise masking and alpha-masking ensure the new asset respects the edges of the original structure, creating a photorealistic composite.

---

## 📸 Visual Journey: Data Trace

The repository contains a complete execution trace:
- **Base Image (`one-window.jpg`):** The original "As-Is" photo of the property.
- **Detection (`prediction.jpg`):** Visualization of the AI identifying the target renovation area.
- **Asset (`midjourney.png`):** The AI-generated modern window design.
- **Output (`modified_image.jpg`):** The final photorealistic renovation.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

### Usage
Run the main pipeline to process the local demonstration assets:
```bash
python3 main.py
```

---

## 🛤 Future Roadmap
- [ ] **Instance Segmentation:** Move from bounding boxes to pixel-perfect segmentation masks for complex shapes.
- [ ] **Automated Prompt Engineering:** Link detection labels to style prompts for fully autonomous renovations.
- [ ] **Web UI:** Build a Streamlit or React frontend for interactive real-time design swapping.

---

## 🤝 Contributing
Contributions are welcome! Whether you are a business strategist looking to refine use cases or a developer optimizing Computer Vision logic, your input is valued.

---

*Developed with ❤️ for the future of PropTech.*
