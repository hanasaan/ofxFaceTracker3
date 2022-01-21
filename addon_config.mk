meta:
	ADDON_NAME = ofxFaceTracker3
	ADDON_DESCRIPTION =
	ADDON_AUTHOR = Yuya Hanai
	ADDON_TAGS = "computer vision"
	ADDON_URL = http://github.com/hanasaan/ofxFaceTracker3

common:
	# dependencies with other addons, a list of them separated by spaces
	# or use += in several lines
	ADDON_DEPENDENCIES = ofxCv ofxOnnxRuntime

	# some addons need resources to be copied to the bin/data folder of the project
	# specify here any files that need to be copied, you can use wildcards like * and ?
	ADDON_DATA = model/
