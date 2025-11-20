# Live Face Recognition

A Python application that detects faces and eyes in real-time using your webcam. Built with OpenCV for efficient face and eye detection.

## Features

- 🎥 Real-time face detection from webcam
- 👁️ Eye detection within detected faces
- ⚡ Threading support for smooth video playback
- 🎯 Reference image comparison
- 📊 Visual feedback with rectangles and text

## Requirements

- Python 3.11+
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd "Live Face Recongnition"
```

2. Install required packages:
```bash
pip install opencv-python numpy
```

## How to Use

1. **Prepare a reference image:**
   - Take a clear photo of the face you want to recognize
   - Save it as `reference.jpg` in the same folder as `main.py`

2. **Run the program:**
```bash
python main.py
```

3. **Using the program:**
   - The webcam will open in a window
   - Blue rectangles show detected faces
   - Green rectangles show detected eyes
   - Green "Match!" text means a face was detected
   - Red "No Match" text means no face was detected
   - Press **Q** to quit the program

## How It Works

The program uses:
- **Haar Cascade Classifiers** - Pre-trained machine learning models for face and eye detection
- **Threading** - Runs face detection in parallel to keep video smooth
- **OpenCV** - Processes video frames and draws visual feedback

## Project Structure

```
Live Face Recongnition/
├── main.py              # Main program file
├── run_project.py       # Alternative run script
├── reference.jpg        # Your reference face image (not included)
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## Notes

- This uses basic face detection, not advanced facial recognition
- The program checks faces every 10 frames to optimize performance
- Make sure your webcam is working before running the program
- For better accuracy, use a clear, well-lit reference image

