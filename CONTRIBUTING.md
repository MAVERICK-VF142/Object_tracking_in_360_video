
# Contributing to YOLOv9 Object Tracking

Thank you for considering contributing to the YOLOv9 Object Tracking project! Contributions are essential for improving this project, and we welcome contributions from everyone. Below is a guide to help you get started.

## How to Contribute

### 1. Reporting Bugs
If you encounter any bugs, please report them by opening an issue. When reporting, please include:
- A clear and descriptive title.
- Steps to reproduce the issue.
- Any error messages or logs, if applicable.
- The environment you are working in (OS, Python version, dependencies).

### 2. Feature Requests
If you have an idea for a new feature or an enhancement, we’d love to hear it! Please submit a feature request by opening an issue and clearly describing:
- The proposed feature.
- The problem it solves or how it enhances the current functionality.
- Any related code snippets or resources (if applicable).

### 3. Forking the Repository
To contribute to this project, first fork the repository and then follow these steps:
1. **Fork the repository**:
   - Go to the [main repository](https://github.com/MAVERICK-VF142/Object_tracking_in_360_video).
   - Click the "Fork" button at the top-right corner.

2. **Clone the repository**:
   - Open your terminal and clone the repository:
   ```bash
   git clone https://github.com/your-username/Object_tracking_in_360_video.git
   cd Object_tracking_in_360_video
   ```

3. **Create a new branch**:
   - For new features or bug fixes, create a new branch with a descriptive name:
   ```bash
   git checkout -b feature-name
   ```

4. **Make your changes**:
   - Implement the desired features or bug fixes, ensuring your code follows best practices and is well-documented.

5. **Run Tests**:
   - Make sure your changes do not break any existing functionality. Run the application and test it by selecting objects in the video, as mentioned in the [README](README.md).

6. **Commit your changes**:
   - After making your changes, commit them with a meaningful message:
   ```bash
   git add .
   git commit -m "Add: [Feature description] or Fix: [Bug description]"
   ```

7. **Push to GitHub**:
   - Push your branch to your forked repository:
   ```bash
   git push origin feature-name
   ```

8. **Create a Pull Request**:
   - Go to the original repository on GitHub and create a pull request (PR) from your branch. Please describe the changes you made and any related issue or feature request it addresses.

### 4. Improving Documentation
If you notice any inaccuracies or missing information in the documentation, feel free to submit a pull request to update or improve it.

### 5. Code Review Process
All pull requests will be reviewed by the maintainers. Feedback will be provided, and you might be requested to make revisions before the PR is merged. Please be patient and address any requested changes in a timely manner.

## Code Style Guidelines
Please ensure your code:
- Follows Python's PEP 8 style guide.
- Is well-documented and includes comments where necessary.
- Includes proper error handling.

## Development Setup
1. Clone the repository as described above.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLOv9 model weights (`yolov9e.pt`) and place them in the root directory.

## License
By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

---

This document serves as a guide for all contributors to ensure smooth collaboration and high-quality code contributions. If you have any questions, feel free to reach out by opening an issue.


