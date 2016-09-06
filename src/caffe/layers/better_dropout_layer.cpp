// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

	// const int count = bottom[0]->count();
	// const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	const int K = channels * height * width;

	fixed_lamda_ = this->layer_param_.better_dropout_param().fixed_lamda();
	fixed_p_ = this->layer_param_.better_dropout_param().fixed_p();
	updata_stride_ = this->layer_param_.better_dropout_param().updata_stride();
	p_lr_ = this->layer_param_.better_dropout_param().p_lr();
	lamda_lr_ = this->layer_param_.better_dropout_param().lamda_lr();

	iteration_ = 0;

	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		this->blobs_.resize(2);
		this->blobs_[0].reset(new Blob<Dtype>(1,1,1,K));
		this->blobs_[1].reset(new Blob<Dtype>(1,1,1,K));

		shared_ptr<Filler<Dtype> > p_filler(GetFiller<Dtype>(
				this->layer_param_.better_dropout_param().p_filler()));
		p_filler->Fill(this->blobs_[0].get());

		shared_ptr<Filler<Dtype> > lamda_filler(GetFiller<Dtype>(
				this->layer_param_.better_dropout_param().lamda_filler()));
		lamda_filler->Fill(this->blobs_[1].get());
	}
}

template <typename Dtype>
void BDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
  theta_.ReshapeLike(*bottom[0]);
  epsilon_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	Dtype* theta_data = theta_.mutable_cpu_data();
	const Dtype* p_data = this->blobs_[0]->cpu_data();
	const Dtype* lamda_data = this->blobs_[1]->cpu_data();
	Dtype* epsilon_data = epsilon_.mutable_cpu_data();

	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	const int K = channels * height * width;
	
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

	if (this->phase_ == TRAIN) {
		// Create theta & epsilon
		caffe_rng_uniform(count, static_cast<Dtype>(0), static_cast<Dtype>(1), theta_data);
		for (int i = 0; i < count; ++i)	{
			int t = i % K;
			theta_data[i] = log (1 - theta_data[i]) / (-lamda_data[t]);
		}

		for (int i = 0; i < count; ++i)	{
			int t = i % K;
			epsilon_data[i] = asinh(p_data[t]/(2 * exp(-lamda_data[t] * bottom_data[i]))) / lamda_data[t];
		}

		for (int i = 0; i < count; ++i)	{
			int t = i % K;
			if ((bottom_data[i] - epsilon_data[i]) < theta_data[i] && (bottom_data[i] + epsilon_data[i]) > theta_data[i])
				top_data[t] = bottom_data[i] / ( 1 - p_data[t]);
			else top_data[t] = 0;
		}
	}	else {
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}
}

template <typename Dtype>
void BDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const int count = bottom[0]->count();
	Dtype* p_data = this->blobs_[0]->mutable_cpu_data();
	Dtype* lamda_data = this->blobs_[1]->mutable_cpu_data();
	const Dtype* epsilon_data = epsilon_.cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* theta_data = theta_.cpu_data();
	if (top[0] == bottom[0]) { bottom_data = bottom_memory_.cpu_data();}
	iteration_++;

	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	const int K = channels * height * width;

	if (!(iteration_ % updata_stride_)) {
		for (int i = 0; i < K; ++i)	{
			LOG(INFO)<<"At pos: "<<i<<"   p = "<<p_data[i]<<"   lamda = "<<lamda_data[i];
		}

		if (!fixed_p_) {
			for (int i = 0; i < count; ++i)	{
				int t = i % K;
				if ((bottom_data[i] - epsilon_data[i]) < theta_data[i] && (bottom_data[i] + epsilon_data[i]) > theta_data[i])
					p_data[t] += (1 / (lamda_data[t] * p_data[t])) * top_diff[i] * p_lr_;
			}
		}

		if (!fixed_lamda_) {
			for (int i = 0; i < count; ++i)	{
				int t = i % K;
				if ((bottom_data[i] - epsilon_data[i]) < theta_data[i] && (bottom_data[i] + epsilon_data[i]) > theta_data[i])
					lamda_data[t] -= ((log(p_data[t]/epsilon_data[i]/2) - log(lamda_data[t]) + 1) / lamda_data[t] / lamda_data[t] 
						+ epsilon_data[i] * epsilon_data[i] / 6) * top_diff[i] * lamda_lr_;
			}
		}
	}

	if (propagate_down[0]) {
		if (this->phase_ == TRAIN) {
			for (int i = 0; i < count; ++i) {
				int t = i % K;
				if ((bottom_data[i] - epsilon_data[i]) < theta_data[i] && (bottom_data[i] + epsilon_data[i]) > theta_data[i])
					bottom_diff[i] = top_diff[i] / ( 1 - p_data[t]);
				else bottom_diff[i] = 0;
			}
		} else {
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(BDropoutLayer);
#endif

INSTANTIATE_CLASS(BDropoutLayer);
REGISTER_LAYER_CLASS(BDropout);

}  // namespace caffe
