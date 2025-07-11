# Live Face Analyzer 🧠

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A real-time face analysis application that detects emotions, age, gender, and race using your webcam feed, powered by DeepFace and Streamlit.

![Demo GIF](assets/demo.gif)

## ✨ Features

- **Real-time Analysis**:
  - Emotion detection (7 basic emotions)
  - Age estimation
  - Gender classification
  - Race prediction
- **Interactive Dashboard**:
  - Live video feed with annotations
  - Dynamic charts and statistics
  - Session history tracking
- **Data Management**:
  - Pause/Resume analysis
  - Export session data to CSV
  - Download project documentation

## 🛠️ Tech Stack

| Component        | Technology |
|-----------------|------------|
| Frontend        | Streamlit |
| Face Analysis   | DeepFace |
| Video Processing| OpenCV |
| Data Handling   | Pandas |
| Visualization   | Matplotlib |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Chrome/Firefox browser

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Prajwal9823/live-face-analyzer.git
cd live-face-analyzer
```
2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
streamlit run app.py
