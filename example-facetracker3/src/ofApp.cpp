#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	// Setup grabber
	if (grabber.setup(1280, 720)) {

		// Setup tracker
#ifdef _MSC_VER
		tracker.setupGpu(); // CUDA
		//tracker.setupGpu("model/yolov5s-face_640x640.onnx", 0, true); // TensorRT
		//tracker.setupCpu(); // CPU
#else
		tracker.setupCpu();
#endif
	}
}

//--------------------------------------------------------------
void ofApp::update(){
	grabber.update();

	// Update tracker when there are new frames
	if (grabber.isFrameNew()) {
		tracker.update(grabber);
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	// Draw camera image
	grabber.draw(0, 0);

	// Draw tracker landmarks
	tracker.drawDebug();

	// Draw text UI
	ofDrawBitmapStringHighlight("Framerate : " + ofToString(ofGetFrameRate()), 10, 40);
	ofDrawBitmapStringHighlight("Tracker thread framerate : " + ofToString(tracker.getThreadFps()), 10, 60);
}
