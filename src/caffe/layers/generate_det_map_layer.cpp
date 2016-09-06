#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GenerateDetMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	if (bottom.size() == 3)
		cropped_ = false;
	else {
		if (bottom.size() == 4)
			cropped_ = true;
		else
			LOG(FATAL) << "Must take 3 or 4 input blobs. Original image, Score map, Detection Annotations, Cropped margin.";
	}
	// CHECK_EQ(bottom.size(), 3) << "Must take 3 input blobs. Original image, Score map, Detection Annotations.";

	if (top.size()>1)	{
		CHECK_EQ(top.size(),3) << "Must output mask and label simultaneously.";
	}
}

template <typename Dtype>
void GenerateDetMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	in_num_ = bottom[0]-> num();
	in_channels_ = bottom[0]-> channels();
	in_height_ = bottom[0]-> height();
	in_width_ = bottom[0]-> width();

	out_num_ = bottom[1]-> num();
	out_channels_ = bottom[1]-> channels();
	out_height_ = bottom[1]-> height();
	out_width_ = bottom[1]-> width();

	top[0]->Reshape(out_num_, out_channels_, out_height_, out_width_);
	if (top.size() == 3) {
		if (this->layer_param_.generate_det_map_param().background_class()) {
			top[1]->Reshape(bottom[2]->channels()+1,out_channels_,out_height_, out_width_);
			top[2]->Reshape(bottom[2]->channels()+1,1,1,1);
		}
		else {
			top[1]->Reshape(bottom[2]->channels(),out_channels_,out_height_, out_width_);
			top[2]->Reshape(bottom[2]->channels(),1,1,1);
		}
	}
}

template <typename Dtype>
void GenerateDetMapLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void GenerateDetMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(GenerateDetMapLayer);
#endif

INSTANTIATE_CLASS(GenerateDetMapLayer);
REGISTER_LAYER_CLASS(GenerateDetMap);

}  // namespace caffe
