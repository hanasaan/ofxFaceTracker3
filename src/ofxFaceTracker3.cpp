#include "ofxFaceTracker3.h"

#include "ofGraphics.h"


namespace ofxFaceTracker3
{
	using namespace ofxOnnxRuntime;

//#define ENABLE_TIME_PROFILE
#ifdef ENABLE_TIME_PROFILE
	class TinyScopedTimeProfiler
	{
		float ts, te;
		std::string name;
	public:
		TinyScopedTimeProfiler(std::string name) {
			ts = ofGetElapsedTimef();
			this->name = name;
		}

		~TinyScopedTimeProfiler() {
			te = ofGetElapsedTimef();
			std::cerr << "[TimeProfile] " << name << ": "<< (te - ts) * 1000 << "ms" << std::endl;
		}
	};
#define DEBUG_TIME_PROFILE(A) auto tp = TinyScopedTimeProfiler(A);
#else
#define DEBUG_TIME_PROFILE(A) 
#endif


	Tracker::Tracker()
	{
	}

	Tracker::~Tracker()
	{
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

		// if input image has 1, 4 channels, convert it to RGB.
		if (mat.channels() == 1) {
			cv::cvtColor(mat, mat_rgb, cv::COLOR_GRAY2RGB);
		}
		else if (mat.channels() == 3) {
			mat_rgb = mat.clone();
		}
		else if (mat.channels() == 4) {
			cv::cvtColor(mat, mat_rgb, cv::COLOR_RGBA2RGB);
		}
		else {
			return false;
		}

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
			mat_blob = cv::dnn::blobFromImage(mat_rgb_padded, 1.0 / 255.0, cv::Size(input_node_dims[2], input_node_dims[3]), cv::Scalar(), false, true);
			std::memcpy(input_values_handler.data(), mat_blob.data, input_tensor_size * sizeof(float));
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
				result.bbox.width *= inv_scale;
				result.bbox.height *= inv_scale;
				for (auto& kp : result.keypoints) {
					kp *= inv_scale;
				}
			}
		}

		detection_frame_result = results_merged;

		return true;
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
			ofDrawBitmapString(ofVAArgsToString("Score : %.3f", result.score), result.bbox.x, result.bbox.y);
		}
		ofPopMatrix();
		ofDrawBitmapStringHighlight(ofVAArgsToString("Detected Face Count : %d", detection_frame_result.size()), x + 10, y + 20);
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

			ofSleepMillis(1);
		}
	}
}