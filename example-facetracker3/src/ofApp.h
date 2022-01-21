#pragma once

#include "ofMain.h"
#include "ofxFaceTracker3.h"

class ofApp : public ofBaseApp{
	ofxFaceTracker3::Tracker tracker;
	ofVideoGrabber grabber;
public:
	void setup();
	void update();
	void draw();
};
