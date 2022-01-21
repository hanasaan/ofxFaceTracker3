# ofxFaceTracker3
**Working in progress**

Fast and robust face tracking addon for openFrameworks based on [YOLO5Face](https://github.com/deepcam-cn/yolov5-face) and [ONNX Runtime](https://github.com/microsoft/onnxruntime).

## Features
- Fast and robust face & keypoints detection using [YOLO5Face](https://github.com/deepcam-cn/yolov5-face).
- Achieve realtime FPS on both CPU and GPU.

## Tested environment
- oF0.11.2 + macOS Catalina Intel CPU
- oF0.11.2 + Windows10 CPU / CUDA / TensorRT
    - CUDA 11.4, TensorRT 8.0.3.4

## Installation
- This addon depends on following addons. Please pull them to `${OF_BASE_PATH}/addons` directory first.
    - [ofxCv](https://github.com/kylemcdonald/ofxCv)
    - [ofxOnnxRuntime](https://github.com/hanasaan/ofxOnnxRuntime)
- Generate project using project generator, then `model` directory is copied into `bin/data`.

## Usage
- In `model` directory, there are 2 converted pretrained models, which are `yolov5s-face_640x640.onnx` and `yolov5n-face0.5_320x320.onnx`.
    - `yolov5s-face_640x640.onnx` is suitable for GPU detection, and `yolov5n-face0.5_320x320.onnx` is suitable for CPU detection with slightly accuracy degradation.
    - Original PyTorch pretrained models can be downloaded from [here](https://github.com/deepcam-cn/yolov5-face#pretrained-models).
    - Then onnx files are generated using [this script](https://github.com/deepcam-cn/yolov5-face/blob/d5d1ad2847142ff37a97a646516aad8655e156ff/export.py).
- `ofxFaceTracker3::Tracker::setupCpu();` is handy setup method for CPU detection, which loads `yolov5n-face0.5_320x320.onnx` by default.
- `ofxFaceTracker3::Tracker::setupGpu();` is handy setup method for GPU detection, which loads `yolov5s-face_640x640.onnx` by default.
- See `example-facetracker3` for more details.

## Notes
- If TensorRT is enabled, it takes long time when starting app for the first time. In my environment, it takes 12 minutes. Then converted `*.trt` file is generated under `bin/data/model/yolov5s-face_640x640_trt_cache` directory.

## Comparison to [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2)
- For easy environment such as no difficult lighting, no occulusion and frontal angle, ofxFaceTracker2 might be better because it runs on CPU with super lightweight load.
- However ofxFaceTracker2 hardly detects masked faces which are common in COVID-19 era, and also it does not support non-frontal faces. 
- If you guys face any of difficult detection conditions, ofxFaceTracker3 will perform lots better than ofxFaceTracker2.

## Reference
- I heavily referred [Lite.AI.ToolKit](https://github.com/DefTruth/lite.ai.toolkit) implementation.
