#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	// Setup grabber
	auto devices = grabber.listDevices();
	for (auto device : devices) {
		cerr << device.id << ": " << device.deviceName << ", available=" << device.bAvailable << ", formatsz=" << device.formats.size() << endl;
		for (auto format : device.formats) {
			cerr << "    " << format.width << "x" << format.height << ", fps=";
			for (auto fps : format.framerates) {
				cerr << fps << ", ";
			}
			cerr << endl;
		}
	}

	grabber.setDeviceID(5);
	if (grabber.setup(2560, 720)) {
	}

	// Setup tracker
	tracker.setupGpu();
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
