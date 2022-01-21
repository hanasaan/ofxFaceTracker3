#pragma once
#include "ofThread.h"
#include "ofThreadChannel.h"
#include "ofxCv.h"
#include "ofxOnnxRuntime.h"

namespace ofxFaceTracker3
{
	static constexpr size_t NUM_KEYPOINTS = 5;
	static constexpr size_t NUM_BUFFERS = 4;

	struct DetectionResult
	{
		ofRectangle bbox;
		std::array<glm::vec2, NUM_KEYPOINTS> keypoints;
		float score;
		unsigned int tracking_label;
	};

	using DetectionFrame = std::vector<DetectionResult>;

	static inline float getIOU(const ofRectangle& a, const ofRectangle& b) {
		auto intersection = a.getIntersection(b).getArea();
		return (intersection / (a.getArea() + b.getArea() - intersection));
	}

	class Tracker : public ofxOnnxRuntime::BaseHandler, public ofThread
	{
	public:
		Tracker();
		~Tracker();

		/// Easy shortcut
		void setupCpu(const std::string& onnx_path = "model/yolov5n-face0.5_320x320.onnx");
		void setupGpu(const std::string& onnx_path = "model/yolov5s-face_640x640.onnx", int device_id = 0, bool tensorrt = false);

		/// Update the trackers input image
		bool update(cv::Mat image, cv::Rect roi = cv::Rect(0, 0, 0, 0));

		template <class T>
		bool update(T& image, cv::Rect roi = cv::Rect(0, 0, 0, 0)) {
			return update(ofxCv::toCv(image), roi);
		}
		void updateThreadedResult();

		/// Draw a debug drawing of the detected face
		void drawDebug(float x = 0, float y = 0) const;

		void drawDebugInformation() const;

		/// Stop the background tracker thread (called automatically on app exit)
		void stop();

		/// Get number of detected faces
		size_t size() const;

		/// Returns the fps the background tracker thread is running with
		float getThreadFps()const;

		/// Set weather the tracker should run threaded or not. Must be called before calling setup().
		void setThreaded(bool threaded);
		bool isThreaded() const { return b_threaded; }

		const DetectionFrame& getDetectionFrameResult() const;
	protected:
		void threadedFunction() override;
		DetectionFrame runDetection(cv::Mat mat_rgb);

		bool b_threaded = true;
		ofFpsCounter thread_fps;

		/// YOLO5face params
		float score_threshold = 0.3f;
		float iou_threshold = 0.45f;
		size_t max_face_count = 400;

		/// all intermediate cv::Mat buffers, to avoid memory allocation
		int last_task_index = 0;
		std::array<cv::Mat, NUM_BUFFERS> mat_task_buffers;
		cv::Mat mat_rgb_resized;
		cv::Mat mat_rgb_padded;
		cv::Mat mat_rgb_padded_f;

		/// result buffer
		DetectionFrame detection_frame_result;

		/// tracker
		ofxCv::Tracker<cv::Rect> face_tracker;

		/// thread channel related
		ofThreadChannel<cv::Mat*> to_process;
		ofThreadChannel<DetectionFrame> processed;
	};
}

