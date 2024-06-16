# Multi-Object Detection and Recognition System

This project implements a multi-threaded system for real-time detection and recognition of various objects using computer vision techniques. It leverages OpenCV for image processing and pthreads for parallel execution on different CPU cores.

## Features

- **Pedestrian Detection:** Detects pedestrians using HOG (Histogram of Oriented Gradients) descriptors.
- **Lane Detection:** Detects lanes on the road using Canny edge detection and Hough transform.
- **Stop Light Detection:** Identifies stop lights based on color segmentation and Hough circles.
- **Stop Sign Detection:** Detects stop signs using color segmentation and Hough circles.
- **Performance Monitoring:** Calculates and logs FPS (Frames Per Second) for each detection module using syslog.

## Requirements

- C++ compiler with C++11 support
- OpenCV 4.0 or higher
- pthreads library

## Installation

Clone the repository and compile the project:

git clone https://github.com/your-username/multi-object-detection.git<br>
cd multi-object-detection<br>
mkdir build && cd build<br>
cmake ..<br>
cmake --build .<br>

## Usage
Run the compiled executable with command-line options to enable different detection modules:

./multi-object-detection --p --l --sl --ss <br>
--p: Enable pedestrian detection.<br>
--l: Enable lane detection.<br>
--sl: Enable stop light detection.<br>
--ss: Enable stop sign detection.<br>

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements, bug fixes, or new features.
