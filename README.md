![Header Image](ppe-public-view.png)
# Construction Safety Detection using YOLOv8

This project focuses on enhancing construction site safety through real-time detection of safety gear such as helmets and vests worn by workers, as well as detecting the presence of a person. The detection is performed using YOLOv8, a state-of-the-art object detection algorithm.

## Overview

Construction sites present various safety hazards, and ensuring that workers wear appropriate safety gear is crucial for accident prevention. This project aims to automate the process of safety gear detection using computer vision techniques. By deploying YOLOv8, the system can detect whether a worker is wearing a helmet, vest, and detect the presence of a person within the construction site premises.

## Features

- **Helmet Detection:** Detects whether a worker is wearing a helmet.
- **Vest Detection:** Detects whether a worker is wearing a safety vest.
- **Person Detection:** Detects the presence of a person within the construction site.

## Requirements

- Python 3.x
- YOLOv8 dependencies (refer to YOLOv8 documentation for installation instructions)
- OpenCV
- Other dependencies as mentioned in the project code

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Ansarimajid/Construction-PPE-Detection.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 weights file and place it in the designated directory.

## Usage

1. Navigate to the project directory.

2. Run the detection script:

    ```bash
    python detect.py
    ```

3. The script will initiate real-time detection using your webcam or process a video file.

4. Detected objects will be highlighted with bounding boxes indicating whether a helmet and/or vest is worn, and if a person is detected.

## Customization

You can fine-tune the detection parameters and thresholds in the `detect.py` script to adapt to different environments and requirements.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is built upon the YOLOv8 architecture developed by [YOLO](https://github.com/AlexeyAB/darknet).
- Special thanks to the contributors and open-source community for their valuable insights and contributions.
