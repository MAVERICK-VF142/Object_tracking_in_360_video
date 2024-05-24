
# YOLOv8 Object Tracking

This repository contains a Python script for real-time object tracking using YOLOv8 and OpenCV. It allows you to select an object in a video frame, track it, and display its coordinates in real-time.

## Requirements

- numpy==1.24.3
- opencv_python==4.8.1.78
- ultralytics

## Installation

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/MAVERICK-VF142/Object_tracking_in_360_video.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv9 model weights (`yolov9e.pt`) and place them in the root directory of this repository.

## Usage

1. Run the script `object_tracking.py`:

   ```
   python object_tracking.py
   ```

2. Select an object in the video frame by clicking on it. The script will track the selected object and display its coordinates in real-time.

3. Press 'q' to quit the application.

---

## Contributing

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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
