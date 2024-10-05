
# 🤖 YOLOv9 Object Tracking

![Repo Stars](https://img.shields.io/github/stars/MAVERICK-VF142/Object_tracking_in_360_video) ![Forks](https://img.shields.io/github/forks/MAVERICK-VF142/Object_tracking_in_360_video) ![Issues](https://img.shields.io/github/issues/MAVERICK-VF142/Object_tracking_in_360_video)

This repository contains a Python script for real-time object tracking using YOLOv9 and OpenCV. It allows you to select an object in a video frame, track it, and display its coordinates in real-time.

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)
- [Contributors](#-our-valuable-contributors-%E2%9D%A4%EF%B8%8F%E2%9C%A8)

## 🚀 Features

- **Real-time object tracking** using YOLOv9 and OpenCV.
- **User-friendly interface** for selecting objects in video frames.
- **Real-time display** of object coordinates.
- **360° video support** for enhanced video experiences.

## 📖 Requirements

    easydict==1.13
    gdown==5.2.0
    ipdb==0.13.13
    motmetrics==1.4.0
    numpy==1.24.4
    opencv_python==4.8.1.78
    pandas==1.5.3
    PyYAML==6.0.1
    scipy==1.11.4
    torch==2.3.0
    torchvision==0.18.0
    ultralytics

## 📦 Installation

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/MAVERICK-VF142/Object_tracking_in_360_video.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv9 model weights (`yolov9e.pt`) and place them in the root directory of this repository.

## ▶️ Usage

1. Run the script `app.py`:

   ```
   python app.py
   ```

2. Select an object in the video frame by clicking on it. The script will track the selected object and display its coordinates in real-time.

3. Press 'q' to quit the application.

## 🛠️ Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository to your GitHub account.

2. Clone the forked repository to your local machine:
   
   ```
   git clone https://github.com/your-username/Object_tracking_in_360_video.git
   ```

3. Create a new branch for your feature or bug fix:
   
   ```
   git checkout -b feature-name
   ```

   Replace `feature-name` with a descriptive name for your feature or bug fix.

4. Make your changes and commit them:
   
   ```
   git add .
   git commit -m "Description of your changes"
   ```

5. Push your changes to your forked repository:
   
   ```
   git push origin feature-name
   ```

6. Create a pull request from your forked repository to the main repository's `master` branch.

   **Note:** Please ensure your pull request adheres to the repository's contribution guidelines.

Thank you for contributing to this project!


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ❤️ Our Valuable Contributors ❤️

[![Contributors](https://contrib.rocks/image?repo=MAVERICK-VF142/Object_tracking_in_360_video)](https://github.com/MAVERICK-VF142/Object_tracking_in_360_video/graphs/contributors)

---
