# 👁️ Universal Access AI

**A Next-Generation Desktop Assistant for the Visually Impaired**

Universal Access AI is an intelligent accessibility tool designed to help users with vision impairments navigate their digital and physical environments independently. By combining **Computer Vision**, **Optical Character Recognition (OCR)**, and **Generative AI**, it acts as a *digital pair of eyes* that can see, read, and converse in real time.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

<img width="1920" height="1017" alt="Universal Access AI Screenshot" src="https://github.com/user-attachments/assets/a1e7aacb-64df-4006-98f5-2a338bf0ed98" />

---

## 🚀 Key Features

- **🗣️ Full Voice Control**  
  Completely hands-free operation. Users can issue voice commands and receive audio feedback.

- **🔍 Object Detection**  
  Locates everyday items (e.g., *“Find my phone”*, *“Where are my keys?”*) and provides spatial guidance.

- **📷 Smart OCR Scanner**  
  Reads text from physical documents, books, and screens instantly.

- **👁️ Scene Description**  
  Analyzes the live camera feed and describes the environment in natural language  
  *(e.g., “I see a person sitting at a desk with a laptop”)*.

- **🤖 AI Companion**  
  Integrated with **Llama 3 (via Groq)** to answer general questions and provide intelligent assistance.

- **⚡ Zero-Latency Mode**  
  Uses multi-threading to ensure camera, voice, and AI modules run smoothly without UI freezing.

---

## 🛠️ Installation Guide

### Prerequisites
- **Python 3.10 or newer**
- **Tesseract OCR** (required for text reading features)  
  https://github.com/UB-Mannheim/tesseract/wiki

---

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Universal-Access-AI.git
cd Universal-Access-AI

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download Required AI Models (Critical Step)

This project uses YOLOv3 for object detection. Due to file size limits, the weights must be downloaded manually.

Download yolov3.weights (~237 MB)

Place the file in the main project directory (same folder as main.py)

Ensure the following files are present:

yolov3.weights

yolov3.cfg

coco.names

4️⃣ Install Tesseract OCR

For the Read Text feature to work:

Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

Default install path:
C:\Program Files\Tesseract-OCR

The application automatically detects this path.

5️⃣ Run the Application
python main.py

🌟 Key Benefits

Accessibility First
Designed specifically for visually impaired users to navigate the world independently.

Cost-Effective
Runs on a standard laptop with a webcam—no expensive hardware required.

Privacy-Focused
Video processing and object detection run locally on the device.

Multimodal Interaction
Combines voice, vision, and text into a single seamless interface.

👨‍💻 Project Built By

Bhavan Kothalanka
B.Tech 3rd Year Student | AI & Data Science Enthusiast

I am a passionate developer focused on leveraging Artificial Intelligence to solve real-world problems. With a strong foundation in Data Science and Computer Vision, I built Universal Access AI to bridge the gap between technology and accessibility. I enjoy exploring how Large Language Models and Edge AI can be combined to create impactful, human-centered software.
