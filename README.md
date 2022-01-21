# ofxFaceTracker3
**Working in progress**
Fast and robust face tracking addon for openFrameworks based on [YOLO5Face](https://github.com/deepcam-cn/yolov5-face) and [ONNX Runtime](https://github.com/microsoft/onnxruntime).

## Features
- Fast and robust face & keypoints detection using [YOLO5Face](https://github.com/deepcam-cn/yolov5-face).
- Achieve realtime FPS on both CPU and GPU.

## Tested environment
- oF0.11.2 + macOS Catalina Intel CPU
- oF0.11.2 + Windows10 CPU / CUDA
    - CUDA 11.4, TensorRT 8.0.3.4

## Installation
- This addon depends on following addons. Please pull them to `addons` directory first.
    - [ofxCv](https://github.com/kylemcdonald/ofxCv)
    - [ofxOnnxRuntime](https://github.com/hanasaan/ofxOnnxRuntime)
- Generate project using project generator.

## Usage
- see `example-facetracker3`

## Comparison to [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2)
- For easy environment such as no difficult lighting, no occulusion and frontal angle, ofxFaceTracker2 might be better because it runs on CPU with super lightweight load.
- However ofxFaceTracker2 hardly detects masked faces which are common in COVID-19 era, and also it does not support non-frontal faces. 
- If you guys face any of difficult detection conditions, ofxFaceTracker3 will perform lots better than ofxFaceTracker2.

## Reference
- I heavily referred [Lite.AI.ToolKit](https://github.com/DefTruth/lite.ai.toolkit) implementation.

## ToDo
- Threading, Tracking, and TensorRT.
