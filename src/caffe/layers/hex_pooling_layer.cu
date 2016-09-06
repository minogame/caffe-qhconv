#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HexMaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, Dtype* const top_data, int* mask, Dtype* top_mask, const float compensation) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int pw_r = index / pooled_width;
    const int ph = pw_r % pooled_height;
    const int c = (pw_r / pooled_height) % channels;
    const int n = pw_r / pooled_height / channels;

    // int hc = -1;
    // int wc = -1;

    // if ((ph>>1)<<1 == ph) {
    //  hc = ph * 2 + 1;
    //  wc = pw * 2 + 1;
    // } else {
    //  hc = ph * 2 + 1;
    //  wc = pw * 2;
    // }

    int hc = ph * 2 + 1;
    int wc = pw * 2 + 1;

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;

    int cpos = hc * width + wc;

    if (bottom_slice[cpos] > maxval) {
      maxidx = cpos; maxval = bottom_slice[maxidx];} 

    if (wc > 0) {
      if (bottom_slice[cpos - 1] > maxval) {
        maxidx = cpos - 1; maxval = bottom_slice[maxidx];}}

    if (wc + 1 < width) {
      if (bottom_slice[cpos + 1] > maxval) {
        maxidx = cpos + 1; maxval = bottom_slice[maxidx];}} 

    if (hc > 0) {
      if (bottom_slice[cpos - width] > maxval) {
        maxidx = cpos - width; maxval = bottom_slice[maxidx];}}

    if (hc + 1 < height) {
      if (bottom_slice[cpos + width] > maxval) {
        maxidx = cpos + width; maxval = bottom_slice[maxidx];}} 

    if (hc > 0 && wc > 0) {
      if (bottom_slice[cpos - width - 1] > maxval) {
        maxidx = cpos - width - 1; maxval = bottom_slice[maxidx];}}

    if (hc + 1 < height && wc > 0) {
      if (bottom_slice[cpos + width - 1] > maxval) {
        maxidx = cpos + width; maxval = bottom_slice[maxidx];}} 

    top_data[index] = maxval * compensation;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void GMaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask, const float compensation) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval * compensation;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void GAvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, const float compensation) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size * compensation;
  }
}

template <typename Dtype>
__global__ void HexAvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int pw_r = index / pooled_width;
    const int ph = pw_r % pooled_height;
    const int c = (pw_r / pooled_height) % channels;
    const int n = pw_r / pooled_height / channels;

    int hc = -1;
    int wc = -1;

    // if ((ph>>1)<<1 == ph) {
      hc = ph * 2 + 1;
      wc = pw * 2 + 1;
    // } else {
    //  hc = ph * 2 + 1;
    //  wc = pw * 2;
    // }

    Dtype aveval = 0;
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;

    int cpos = hc * width + wc;

    aveval += bottom_slice[cpos];

    if (wc > 0) aveval += bottom_slice[cpos - 1];

    if (wc + 1 < width) aveval += bottom_slice[cpos + 1];

    if (hc > 0) aveval += bottom_slice[cpos - width];

    if (hc + 1 < height) aveval += bottom_slice[cpos + width];

    if (hc > 0 && wc > 0) aveval += bottom_slice[cpos - width - 1];

    if (hc + 1 < height && wc > 0) aveval += bottom_slice[cpos + width - 1];

    top_data[index] = aveval / 7;
  }
}

template <typename Dtype>
void HexPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  switch (this->layer_param_.hex_pooling_param().pool()) {
  case HexPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }

    HexMaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, top_data,
        mask, top_mask, compensation_);
    break;

  case HexPoolingParameter_PoolMethod_AVE:
    HexAvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, top_data);
    break;

  case HexPoolingParameter_PoolMethod_GAVE:
    GAvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, compensation_);
    break;

  case HexPoolingParameter_PoolMethod_GMAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }

    GMaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask, compensation_);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
  
  iterations_++;
  if (iterations_%step_ == 0) compensation_*=c_gamma_;
}


template <typename Dtype>
__global__ void HexMaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* const bottom_diff, const float compensation) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;

    int pos_type = -1;
    int pw = -1;
    int ph = -1;
    switch (h%2) {
      case 0:
        pw = w>>1;
        ph = h>>1;
        if ((w>>1)<<1 == w) pos_type = 0;
        else pos_type = 1;  break;
      case 1:
        pw = w>>1;
        ph = h>>1;
        if ((w>>1)<<1 == w) pos_type = 2;
        else pos_type = 3;  break;
      /*case 2:
        pw = (w+1)>>1;
        ph = h>>1;
        if ((w>>1)<<1 == w) pos_type = 1;
        else pos_type = 2;  break;
      case 3:
        pw = (w+1)>>1;
        ph = h>>1;
        if ((w>>1)<<1 == w) pos_type = 3;
        else pos_type = 2;  break;*/
    }

    const int pos = h * width + w;

    switch (pos_type) {
      case 0:
        if (mask) {
          const int* const mask_slice = mask + offset;
          if (pw < pooled_width && ph < pooled_height)
            if (mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
          if (pw > 0 && ph > 0)
            if (mask_slice[(ph - 1) * pooled_width + pw - 1] == pos)
              gradient += top_diff_slice[(ph - 1) * pooled_width + pw - 1];
        } else {
          const Dtype* const top_mask_slice = top_mask + offset;        
          if (pw < pooled_width && ph < pooled_height)
            if (top_mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
          if (pw > 0 && ph > 0)
            if (top_mask_slice[(ph - 1) * pooled_width + pw - 1] == pos)
              gradient += top_diff_slice[(ph - 1) * pooled_width + pw - 1];
        }
        break;

      case 1:
        if (mask) {
          const int* const mask_slice = mask + offset;
          if (pw < pooled_width && ph < pooled_height)
            if (mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
          if (ph > 0)
            if (mask_slice[(ph - 1) * pooled_width + pw] == pos)
              gradient += top_diff_slice[(ph - 1) * pooled_width + pw];
        } else {  
          const Dtype* const top_mask_slice = top_mask + offset;          
          if (pw < pooled_width && ph < pooled_height)
            if (top_mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
          if (ph > 0)
            if (top_mask_slice[(ph - 1) * pooled_width + pw] == pos)
              gradient += top_diff_slice[(ph - 1) * pooled_width + pw];
        }
        break;

      case 2:
        if (mask) {
          const int* const mask_slice = mask + offset;
          if (pw < pooled_width && ph < pooled_height)
            if (mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
          if (pw > 0)
            if (mask_slice[ph * pooled_width + pw - 1] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw - 1];
        } else {
          const Dtype* const top_mask_slice = top_mask + offset;  
          if (pw < pooled_width && ph < pooled_height)
            if (top_mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
          if (pw > 0)
            if (top_mask_slice[ph * pooled_width + pw - 1] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw - 1];
        }
        break;

      case 3:
        if (mask) {
          const int* const mask_slice = mask + offset;
          if (pw < pooled_width && ph < pooled_height)
            if (mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
        } else {  
          const Dtype* const top_mask_slice = top_mask + offset;          
          if (pw < pooled_width && ph < pooled_height)
            if (top_mask_slice[ph * pooled_width + pw] == pos)
              gradient += top_diff_slice[ph * pooled_width + pw];
        }
        break;
    }

    bottom_diff[index] = gradient * compensation;
  }
}

template <typename Dtype>
__global__ void HexAvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        //int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / 7;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void GAvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff, const float compensation) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient * compensation;
  }
}

template <typename Dtype>
__global__ void GMaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff, const float compensation) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient * compensation;
  }
}



template <typename Dtype>
void HexPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.hex_pooling_param().pool()) {
  case HexPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    HexMaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, bottom_diff, compensation_);
    break;
  case HexPoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    HexAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case HexPoolingParameter_PoolMethod_GAVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    GAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff, compensation_);
    break;
  case HexPoolingParameter_PoolMethod_GMAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    GMaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff, compensation_);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(HexPoolingLayer);


}  // namespace caffe
