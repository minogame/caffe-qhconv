#include <vector>
#include <iostream>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

	kernel_size_ = this->layer_param_.shared_weights_local_param().kernel_size();
	stride_ = this->layer_param_.shared_weights_local_param().stride();
	pad_ = this->layer_param_.shared_weights_local_param().pad();
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	num_output_ = this->layer_param_.shared_weights_local_param().num_output();

	CHECK_GE(kernel_size_, stride_) << "kernel size smaller than stride, no overlapping.";

	height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
	width_out_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

	num_kernel_ = kernel_size_ - stride_ + 1;
	M_ = num_output_;
	K_ = channels_ * kernel_size_ * kernel_size_;
	N_ = height_out_ * width_out_;
	S_ = num_kernel_ * num_kernel_;

	RNK_ = 1.0f / (N_*K_);
	RK_ = 1.0f / (K_);
	RL_ = 1.0f / num_kernel_;
	RWK_ = 1.0f / (width_out_ * K_);
	RN_ = 1.0f / N_;
	RSK_ = 1.0f/ (S_ * K_);

	CHECK_GT(num_output_, 0); 
	CHECK_GE(height_, kernel_size_) << "height smaller than kernel size";
	CHECK_GE(width_, kernel_size_) << "width smaller than kernel size";
	// Set the parameters
	bias_term_ = this->layer_param_.shared_weights_local_param().bias_term();

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (bias_term_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Intialize the weight
		this->blobs_[0].reset(new Blob<Dtype>(
				num_output_, 1, S_, K_));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.shared_weights_local_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, intiialize and fill the bias term
		if (bias_term_) {
			this->blobs_[1].reset(new Blob<Dtype>(M_, 1, S_, 1));
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.shared_weights_local_param().bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());  
		}
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
		" weights.";
	// TODO: generalize to handle inputs of different shapes.
	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
		CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
		CHECK_EQ(channels_, bottom[bottom_id]->channels())
				<< "Inputs must have same channels.";
		CHECK_EQ(height_, bottom[bottom_id]->height())
				<< "Inputs must have same height.";
		CHECK_EQ(width_, bottom[bottom_id]->width())
				<< "Inputs must have same width.";
	}

	// Shape the tops.
	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
	}

	// The im2col result buffer would only hold one image at a time to avoid
	// overly large memory usage.
	col_buffer_.Reshape(
			1, channels_ * kernel_size_ * kernel_size_, height_out_, width_out_);

	weight_buffer_.Reshape(num_output_,1,N_,K_);
	bias_buffer_.Reshape(1,1,M_,N_);

	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

	Dtype* x_data = col_buffer_.mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* bias = this->blobs_[1]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	Dtype* w_data = weight_buffer_.mutable_cpu_data();
	Dtype* b_data = bias_buffer_.mutable_cpu_data();

	//Fill buffer weights with correct weights
	for (int n = 0; n < num_output_; n++){
		for (int r = 0; r < N_; r++){
			int _h_ = r / width_out_;
			int _w_ = r % width_out_;
			int _hk_ = _h_ % num_kernel_;
			int _wk_ = _w_ % num_kernel_;
			caffe_copy(K_,weight + this->blobs_[0]->offset(n,0,_hk_*num_kernel_+_wk_),
									w_data + weight_buffer_.offset(n,0,_h_*width_out_+_w_)); //NEED CHECK
		}
	}

		// 	for (int i = 0; i < 2000; i+= 50)
		// {
		// 	std::cout<<i<<" = "<<*(w_data + i)<<std::endl;
		// }

	if (bias_term_){
		for (int n = 0; n < num_output_; n++){
			for (int r = 0; r < N_; r++){
				int _h_ = r / width_out_;
				int _w_ = r % width_out_;
				int _hk_ = _h_ % num_kernel_;
				int _wk_ = _w_ % num_kernel_;
				caffe_copy(1,bias + this->blobs_[1]->offset(n,0,_hk_*num_kernel_+_wk_),
										b_data + bias_buffer_.offset(0,0,n,_h_*width_out_+_w_)); //NEED CHECK
			}
		}
	}

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	FillerParameter filler_param;
	filler_param.set_value(1);
	ConstantFiller<Dtype> filler(filler_param);
	filler.Fill(&E);

	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, N_, K_);

	Blob<Dtype> slow;
	slow.Reshape(1, 1, N_, K_);
	Dtype* slow_data = slow.mutable_cpu_data();

	for (int n=0; n<num_; n++) {
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
							 width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, x_data);

		for (int nn = 0; nn<N_; nn++){
			for (int k = 0; k<N_; k++){
				slow_data[nn*K_ + k] = x_data[k*N_ + nn];
			}
		}

		for (int m=0; m<num_output_; m++) { 
			caffe_mul(K_*N_, slow_data, w_data + this->weight_buffer_.offset(m),
								intermediate.mutable_cpu_data());

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, 1,  K_,
														(Dtype)1., intermediate.cpu_data(), E.cpu_data(),
														(Dtype)0., top_data + top[0]->offset(n, m));
		}

		if (bias_term_) {
			caffe_add(M_ * N_, b_data,
								top_data + top[0]->offset(n),
								top_data + top[0]->offset(n));
		}
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* x_data = col_buffer_.mutable_cpu_data();
	Dtype* x_diff = col_buffer_.mutable_cpu_diff();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = NULL;

	Dtype* w_data = weight_buffer_.mutable_cpu_data();

	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, 1, K_);

	Blob<Dtype> xt;
	xt.Reshape(1, 1, N_, K_);
	Dtype* xt_data = xt.mutable_cpu_data();

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	Dtype* e_data = E.mutable_cpu_data();

	Blob<Dtype> F;
	F.Reshape(1, 1, 1, N_);
	Dtype* f_data = F.mutable_cpu_data();

	if (bias_term_) {
		bias_diff = this->blobs_[1]->mutable_cpu_diff();
		memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
		for (int n = 0; n < num_; ++n) {
			for (int m = 0; m < M_; m++){
				for (int r = 0; r < N_; r++){
					int _h_ = r / width_out_;
					int _w_ = r % width_out_;
					int _hk_ = _h_ % num_kernel_;
					int _wk_ = _w_ % num_kernel_;			
					caffe_add(1, bias_diff + this->blobs_[1]->offset(m,0,_hk_*num_kernel_+_wk_),
									top_diff + top[0]->offset(n,m) + _h_*width_out_+_w_,
									bias_diff + this->blobs_[1]->offset(m,0,_hk_*num_kernel_+_wk_));
				}
			}
		}
	}

	for (int n = 0; n < num_output_; n++){
		for (int r = 0; r < N_; r++){
			int _h_ = r / width_out_;
			int _w_ = r % width_out_;
			int _hk_ = _h_ % num_kernel_;
			int _wk_ = _w_ % num_kernel_;
			caffe_copy(K_,weight + this->blobs_[0]->offset(n,0,_hk_*num_kernel_+_wk_),
									w_data + weight_buffer_.offset(n,0,_h_*width_out_+_w_)); //NEED CHECK
		}
	}

	memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
	for (int n=0; n<num_; n++) {
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
							 width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, x_data);

		// gradient wrt weight
		for (int m=0; m<num_output_; m++) {
			for (int k=0; k<K_; k++) {
				caffe_mul(N_, top_diff+top[0]->offset(n, m),  
									x_data+col_buffer_.offset(0,k), f_data);
				for (int r=0; r<N_; r++) {
					xt_data[r*K_+k] = f_data[r];
				}
			}

			for (int r = 0; r < N_; r++){
				int _h_ = r / width_out_;
				int _w_ = r % width_out_;
				int _hk_ = _h_ % num_kernel_;
				int _wk_ = _w_ % num_kernel_;
				caffe_cpu_axpby(K_, Dtype(1.0), xt_data + xt.offset(0,0,_h_*width_out_+_w_),
										 Dtype(1.0), weight_diff + this->blobs_[0]->offset(m,0,_hk_*num_kernel_+_wk_));
			}
		}

		// gradient wrt bottom data
		if (propagate_down[0]) {
			memset(x_diff, 0, col_buffer_.count() * sizeof(Dtype));
			for (int m=0; m<num_output_; m++) {
				for (int p = 0; p < N_; p++)	{
					FillerParameter filler_param;
					//Dtype* spe_top_diff_ = (top[0]->mutable_cpu_diff() + top[0]->offset(n, m, p));
					filler_param.set_value(*(top[0]->cpu_diff() + top[0]->offset(n, m) + p));
					ConstantFiller<Dtype> filler(filler_param);
					filler.Fill(&E);
					
					caffe_mul(K_, e_data,
										w_data+weight_buffer_.offset(m,0,p),
										intermediate.mutable_cpu_data());

					caffe_cpu_axpby(K_, Dtype(1.0),
													intermediate.cpu_data(), Dtype(1.0),
													x_diff+col_buffer_.offset(0,p));
				}
			}

			// col2im back to the data
			col2im_cpu(x_diff, channels_, height_, width_, kernel_size_, kernel_size_,
								 pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));

		}
	}

}
/*

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";

	kernel_size_ = this->layer_param_.shared_weights_local_param().kernel_size();
	stride_ = this->layer_param_.shared_weights_local_param().stride();
	pad_ = this->layer_param_.shared_weights_local_param().pad();
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	num_output_ = this->layer_param_.shared_weights_local_param().num_output();

	height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
	width_out_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

	M_ = num_output_;
	K_ = channels_ * kernel_size_ * kernel_size_;
	N_ = height_out_ * width_out_;

	CHECK_GT(num_output_, 0); 
	CHECK_GE(height_, kernel_size_) << "height smaller than kernel size";
	CHECK_GE(width_, kernel_size_) << "width smaller than kernel size";
	// Set the parameters
	bias_term_ = this->layer_param_.shared_weights_local_param().bias_term();

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (bias_term_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Intialize the weight
		this->blobs_[0].reset(new Blob<Dtype>(
				num_output_, 1, K_, N_));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.shared_weights_local_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, intiialize and fill the bias term
		if (bias_term_) {
			this->blobs_[1].reset(new Blob<Dtype>(1, 1, M_, N_));
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.shared_weights_local_param().bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());  
		}
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
		" weights.";
	// TODO: generalize to handle inputs of different shapes.
	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
		CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
		CHECK_EQ(channels_, bottom[bottom_id]->channels())
				<< "Inputs must have same channels.";
		CHECK_EQ(height_, bottom[bottom_id]->height())
				<< "Inputs must have same height.";
		CHECK_EQ(width_, bottom[bottom_id]->width())
				<< "Inputs must have same width.";
	}

	// Shape the tops.
	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
	}

	// The im2col result buffer would only hold one image at a time to avoid
	// overly large memory usage.
	col_buffer_.Reshape(
			1, channels_ * kernel_size_ * kernel_size_, height_out_, width_out_);

	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

	Dtype* x_data = col_buffer_.mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	FillerParameter filler_param;
	filler_param.set_value(1);
	ConstantFiller<Dtype> filler(filler_param);
	filler.Fill(&E);

	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, K_, N_);
	for (int n=0; n<num_; n++) {
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
							 width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, x_data);

		for (int m=0; m<num_output_; m++) { 
			caffe_mul(K_*N_, x_data, weight+this->blobs_[0]->offset(m),
								intermediate.mutable_cpu_data());

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
														(Dtype)1., E.cpu_data(),
														intermediate.cpu_data(),
														(Dtype)0., top_data + top[0]->offset(n, m));
		}

		if (bias_term_) {
			caffe_add(M_ * N_, this->blobs_[1]->cpu_data(),
								top_data + top[0]->offset(n),
								top_data + top[0]->offset(n));
		}
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* x_data = col_buffer_.mutable_cpu_data();
	Dtype* x_diff = col_buffer_.mutable_cpu_diff();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* bias_diff = NULL;

	Blob<Dtype> intermediate;
	intermediate.Reshape(1, 1, 1, N_);

	Blob<Dtype> xt;
	xt.Reshape(1, 1, K_, N_);
	Dtype* xt_data = xt.mutable_cpu_data();

	if (bias_term_) {
		bias_diff = this->blobs_[1]->mutable_cpu_diff();
		memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
		for (int n = 0; n < num_; ++n) {
			caffe_add(M_ * N_, bias_diff,
								top_diff + top[0]->offset(n),
								bias_diff);
		}
	}

	memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
	for (int n=0; n<num_; n++) {
		im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
							 width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, x_data);

		// gradient wrt weight
		for (int m=0; m<num_output_; m++) {
			Dtype* filter_weight_diff = weight_diff+this->blobs_[0]->offset(m);
			for (int k=0; k<K_; k++) {
				caffe_mul(N_, top_diff+top[0]->offset(n, m),  
									x_data+col_buffer_.offset(0,k), xt_data+xt.offset(0,0,k));
			}
			caffe_cpu_axpby(K_*N_, Dtype(1.0), xt_data, Dtype(1.0), filter_weight_diff);
		}
			
		// gradient wrt bottom data
		if (propagate_down[0]) {
			memset(x_diff, 0, col_buffer_.count() * sizeof(Dtype));
			for (int m=0; m<num_output_; m++) {
				for (int k=0; k<K_; k++) {
					caffe_mul(N_, top_diff+top[0]->offset(n, m),
										weight+this->blobs_[0]->offset(m,0,k),
										intermediate.mutable_cpu_data());

					caffe_cpu_axpby(N_, Dtype(1.0),
													intermediate.cpu_data(), Dtype(1.0),
													x_diff+col_buffer_.offset(0,k));
				}
			}

			// col2im back to the data
			col2im_cpu(x_diff, channels_, height_, width_, kernel_size_, kernel_size_,
								 pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));

		}
	}

}

*/
#ifdef CPU_ONLY
STUB_GPU(SharedWeightsLocalLayer);
#endif

INSTANTIATE_CLASS(SharedWeightsLocalLayer);
REGISTER_LAYER_CLASS(SharedWeightsLocal);

}  // namespace caffe