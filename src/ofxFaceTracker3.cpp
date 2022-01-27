#include "ofxFaceTracker3.h"

#include "ofGraphics.h"


namespace ofxFaceTracker3
{
	using namespace ofxOnnxRuntime;

//#define ENABLE_TIME_PROFILE
#ifdef ENABLE_TIME_PROFILE
	class TimeProfiler
	{
		std::mutex mtx;
		std::map<std::string, float> tmap;
	public:
		TimeProfiler() {}
		~TimeProfiler() {}

		void addTime(std::string key, float t)
		{
			std::lock_guard<std::mutex> lock(mtx);
			tmap[key] = t;
		}

		std::string getDebugString()
		{
			std::stringstream ss;
			ss << "Profiled Time Info" << std::endl;
			ss << std::fixed << std::setprecision(3);
			{
				std::lock_guard<std::mutex> lock(mtx);
				for (auto& m : tmap) {
					ss << m.first << " : " << m.second * 1000.0 << "msec" << std::endl;
				}
			}
			return ss.str();
		}
	};
	static std::shared_ptr<TimeProfiler> tp_;

	class TinyScopedTimeProfiler
	{
		TimeProfiler* ptr = nullptr;
		float ts, te;
		std::string name;
	public:
		TinyScopedTimeProfiler(std::string name, TimeProfiler* p) {
			ts = ofGetElapsedTimef();
			this->name = name;
			this->ptr = p;
		}

		~TinyScopedTimeProfiler() {
			te = ofGetElapsedTimef();
			ptr->addTime(name, (te - ts));
		}
	};
#define DEBUG_TIME_PROFILE(A) auto tp = TinyScopedTimeProfiler(A, tp_.get());
#else
#define DEBUG_TIME_PROFILE(A) 
#endif


	Tracker::Tracker()
	{
#ifdef ENABLE_TIME_PROFILE
		tp_ = std::make_shared<TimeProfiler>();
#endif
	}

	Tracker::~Tracker()
	{
		stop();
		to_process.close();
		processed.close();
	}

	void Tracker::setupCpu(const std::string& onnx_path)
	{
		BaseHandler::setup(onnx_path);
		if (b_threaded) {
			startThread();
		}
	}

	void Tracker::setupGpu(const std::string & onnx_path, int device_id, bool tensorrt)
	{
		BaseHandler::setup(onnx_path, BaseSetting{ tensorrt ? INFER_TENSORRT : INFER_CUDA, device_id });
		if (b_threaded) {
			startThread();
		}
	}

	bool Tracker::update(cv::Mat image, cv::Rect roi)
	{
		if (image.empty()) return false;
		cv::Mat mat = (roi.width != 0 && roi.height != 0) ? image(roi) : image;

		auto& mat_rgb = mat_task_buffers[last_task_index];
		last_task_index = (last_task_index + 1) % NUM_BUFFERS;

		// if input image has 1, 4 channels, convert it to RGB.
		if (mat.channels() == 1) {
			cv::cvtColor(mat, mat_rgb, cv::COLOR_GRAY2RGB);
		}
		else if (mat.channels() == 3) {
			mat_rgb = b_threaded ? mat.clone() : mat;
		}
		else if (mat.channels() == 4) {
			cv::cvtColor(mat, mat_rgb, cv::COLOR_RGBA2RGB);
		}
		else {
			return false;
		}

		if (b_threaded) {
			to_process.send(DetectionTask{ &mat_rgb, glm::vec2(roi.x, roi.y) });
			updateThreadedResult();
		} else {
			detection_frame_result = runDetection(mat_rgb, glm::vec2(roi.x, roi.y));
		}
		return true;
	}

	void Tracker::updateThreadedResult()
	{
		if (b_threaded) {
			// get latest result
			while (processed.tryReceive(detection_frame_result)) {}
		}
	}

	void Tracker::drawDebug(float x, float y) const
	{
		ofPushMatrix();
		ofTranslate(x, y);
		for (const auto& result : detection_frame_result) {
			ofPushStyle();
			ofNoFill();
			ofSetColor(255, 0, 0);
			ofDrawRectangle(result.bbox);
			ofSetColor(0, 255, 0);
			ofFill();
			for (const auto& kp : result.keypoints) {
				ofDrawCircle(kp, 3);
			}
			ofPopStyle();
			ofDrawBitmapString(ofVAArgsToString("%d, %.3f", result.tracking_label, result.score), result.bbox.x, result.bbox.y);
		}
		ofPopMatrix();
	}

	void Tracker::drawDebugInformation() const
	{
		ofDrawBitmapStringHighlight(ofVAArgsToString("Detected Face Count : %d", detection_frame_result.size()), 10, 20);
#ifdef ENABLE_TIME_PROFILE
		ofDrawBitmapStringHighlight(tp_->getDebugString(), 10, 40);
#endif
	}

	void Tracker::stop()
	{
		waitForThread();
	}

	size_t Tracker::size() const
	{
		return detection_frame_result.size();
	}

	float Tracker::getThreadFps() const
	{
		return thread_fps.getFps();
	}

	void Tracker::setThreaded(bool threaded)
	{
		b_threaded = threaded;
	}

	const DetectionFrame & Tracker::getDetectionFrameResult() const
	{
		return detection_frame_result;
	}

	void Tracker::threadedFunction()
	{
		while (isThreadRunning()) {
			DetectionTask dt;
			while (to_process.tryReceive(dt)) {
				// process only latest frame
				if (!to_process.empty()) {
					continue;
				}
				thread_fps.newFrame();
				auto ret = runDetection(*dt.ptr, dt.offset);
				processed.send(ret);
			}
			ofSleepMillis(1);
		}
	}

	DetectionFrame Tracker::runDetection(cv::Mat mat_rgb, const glm::vec2& offset)
	{
		// perform resize & padding first.
		double scale = 1.0;
		{
			DEBUG_TIME_PROFILE("1-Preprocess");
			scale = std::min<double>((double)input_node_dims[2] / mat_rgb.cols, (double)input_node_dims[3] / mat_rgb.rows);
			cv::resize(mat_rgb, mat_rgb_resized, cv::Size(), scale, scale);
			int pad_right = input_node_dims[2] - mat_rgb_resized.cols;
			int pad_bottom = input_node_dims[3] - mat_rgb_resized.rows;
			cv::copyMakeBorder(mat_rgb_resized, mat_rgb_padded, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT);

			// create blob image & copy to tensor buffer
			// cv::dnn::blobFromImage is slow like 6msec. 
			// this conversion is like 1msec.
			mat_rgb_padded.convertTo(mat_rgb_padded_f, CV_32F, 1.0 / 255.0);
			cv::Mat ch[3];
			int sz = mat_rgb_padded_f.rows * mat_rgb_padded_f.cols;
			for (int i = 0; i < 3; ++i) {
				ch[i] = cv::Mat(mat_rgb_padded_f.rows, mat_rgb_padded_f.cols, CV_32F, &input_values_handler[sz * i]);
			}
			cv::split(mat_rgb_padded_f, ch);
		}

		// run inference
		float *output_ptr = nullptr;
		{
			DEBUG_TIME_PROFILE("2-Inference");
			auto& output = BaseHandler::run();
			output_ptr = output.GetTensorMutableData<float>();
		}

		DetectionFrame results, results_merged;
		{
			// generate bbox
			DEBUG_TIME_PROFILE("3-Postprocess");
			auto output_dims = output_node_dims.at(0); // (1,n,16)
			const unsigned int num_anchors = output_dims.at(1); // n = ?

			unsigned int count = 0;
			for (unsigned int i = 0; i < num_anchors; ++i) {
				const float *row_ptr = output_ptr + i * 16;
				float obj_conf = row_ptr[4];
				if (obj_conf < score_threshold) continue; // filter first.
				float cls_conf = row_ptr[15];
				if (cls_conf < score_threshold) continue; // face score.

				// bounding box
				const float *offsets = row_ptr;
				float cx = offsets[0];
				float cy = offsets[1];
				float w = offsets[2];
				float h = offsets[3];

				auto result = DetectionResult{ ofRectangle(cx - 0.5*w,cy - 0.5*h,w,h),{},obj_conf*cls_conf };

				// keypoints
				const float *kps_offsets = row_ptr + 5;
				for (unsigned int j = 0; j < NUM_KEYPOINTS; ++j) {
					result.keypoints[j] = glm::vec2(kps_offsets[2 * j], kps_offsets[2 * j + 1]);
				}

				results.emplace_back(result);

				count += 1; // limit boxes for nms.
				if (count > max_face_count)
					break;
			}

			// merge & rescale to original image size
			std::sort(results.begin(), results.end(), [](const DetectionResult &a, const DetectionResult &b) { return a.score > b.score; });
			const unsigned int box_num = results.size();
			std::vector<int> merged(box_num, 0);
			for (unsigned int i = 0; i < box_num; ++i)
			{
				if (merged[i]) continue;
				DetectionFrame buf = { results[i] };
				merged[i] = 1;

				for (unsigned int j = i + 1; j < box_num; ++j) {
					if (merged[j]) continue;

					if (getIOU(results[i].bbox, results[j].bbox) > iou_threshold) {
						merged[j] = 1;
					}
				}
				results_merged.emplace_back(results[i]);
			}
			auto inv_scale = 1.0f / scale;
			for (auto& result : results_merged) {
				result.bbox.x *= inv_scale;
				result.bbox.y *= inv_scale;
				result.bbox.x += offset.x;
				result.bbox.y += offset.y;
				result.bbox.width *= inv_scale;
				result.bbox.height *= inv_scale;
				for (auto& kp : result.keypoints) {
					kp *= inv_scale;
					kp += offset;
				}
			}

			// get tracking label
			std::vector<cv::Rect> rects;
			for (auto& r : results_merged) {
				rects.emplace_back(ofxCv::toCv(r.bbox));
			}
			face_tracker.track(rects);
			for (int i = 0; i < results_merged.size(); ++i) {
				results_merged[i].tracking_label = face_tracker.getLabelFromIndex(i);
			}
		}
		return results_merged;
	}
}
