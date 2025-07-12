# Know Who's More Beautiful

A Streamlit-based application that compares two face images and computes a composite attractiveness score using facial geometry, skin tone, jawline, eye shape & color, and hair attributes.

---

## 🚀 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Technology Stack](#technology-stack)
* [Installation](#installation)
* [Tutorial](#tutorial)

  * [Running the App](#running-the-app)
  * [Uploading Images](#uploading-images)
  * [Interpreting Results](#interpreting-results)
* [How It Works](#how-it-works)
* [Contributing](#contributing)
* [License](#license)

---

## 📝 Overview

"Know Who's More Beautiful" is a web application built with Streamlit that allows users to upload two face images and computes a comparative attractiveness score for each. The app leverages facial landmarks, computer vision, and color analysis to provide a detailed breakdown of several metrics.

---

## ✨ Features

* **Face Shape Analysis**: Uses MediaPipe Face Mesh to derive geometric ratios.
* **Skin Tone Evaluation**: Calculates average skin coloration within face region.
* **Jawline Rating**: Combines symmetry and sharpness metrics.
* **Eye Shape & Color**: Assesses eye aspect ratio and estimates dominant iris color.
* **Hair Characteristics**: Estimates color score, density, and baldness.
* **Real-time Progress**: Displays a progress bar with step-by-step status.
* **Visual Comparison**: Highlights the "Hott One" with a stamped winner image.

---

## 🛠 Technology Stack

* **Python 3.8+**
* **Streamlit** for the web interface
* **OpenCV** for image processing
* **MediaPipe** for facial landmark detection
* **NumPy** for numerical computations
* **Pillow** for image annotation

---

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/know-whos-more-beautiful.git
   cd know-whos-more-beautiful
   ```
2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🎓 Tutorial

### Running the App

1. Ensure your virtual environment is activated.
2. Launch Streamlit:

   ```bash
   streamlit run G3.py
   ```
3. A browser window will open at `http://localhost:8501`.

### Uploading Images

* On the main page, you will see two file upload widgets labeled **Face 1** and **Face 2**.
* Click **Browse** and select two JPEG or PNG images.
* Once both images are uploaded, the analysis begins automatically.

### Interpreting Results

* A progress bar tracks each analysis step (face shape → skin → jawline → eyes → hair).
* **Final Scores** are displayed side by side with a detailed breakdown available under each **"Detailed Scores"** expander.
* The image with the higher score is stamped **Hott One** at the bottom.

---

## ⚙️ How It Works

1. **Face Mesh Detection**: MediaPipe identifies 468 facial landmarks.
2. **Metric Calculations**:

   * **Geometric Ratios**: Aspect and cheek/jaw ratios for face shape.
   * **Color Metrics**: Average BGR sampling for skin and hair.
   * **Feature Scores**: Custom formulas for jawline symmetry, eye aspect ratio, and hair density.
3. **Composite Score**: Weighted sum of individual metrics.
4. **Winner Annotation**: Pillow draws a label on the image with the higher score.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
