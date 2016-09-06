#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/util/math_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
				"The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
				" a leveldb/lmdb\n"
				"Usage:\n"
				"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 2 || argc > 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
		return 1;
	}

	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[1], db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	BlobProto sum_blob;
	int count = 0;
	// load first datum
	Datum datum;
	datum.ParseFromString(cursor->value());

	if (DecodeDatumNative(&datum)) {
		LOG(INFO) << "Decoding Datum";
	}

	sum_blob.set_num(1);
	sum_blob.set_channels(1);
	sum_blob.set_height(3);
	sum_blob.set_width(3);


	// sum_blob.set_num(1);
	// sum_blob.set_channels(datum.channels());
	// sum_blob.set_height(datum.height());
	// sum_blob.set_width(datum.width());
	const int data_size = datum.height() * datum.width();
	int size_in_datum = std::max<int>(datum.data().size(),
																		datum.float_data_size());
	for (int i = 0; i < 9; ++i) {
		sum_blob.add_data(0.);
	}
	LOG(INFO) << "Starting Iteration";
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		DecodeDatumNative(&datum);

		const std::string& data = datum.data();
		size_in_datum = std::max<int>(datum.data().size(),
				datum.float_data_size());
		// CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
		// 		size_in_datum;

		if (data.size() != 0) {
			// CHECK_EQ(data.size(), size_in_datum);
			for (int h = 0; h < 3; ++h) {
				for (int w = 0; w < 3; ++w) {
					int pos = h*3 + w;
					float cov = 0;
					for (int i = 0; i < data_size; ++i) {
						int hh = h * data_size + i;
						int ww = w * data_size + i;
						cov += (uint8_t)(data[hh]) * (uint8_t)(data[ww]);
					}
					cov/=data_size;
					sum_blob.set_data(pos, sum_blob.data(pos) + cov);
				}
			}
		} else {
			for (int h = 0; h < 3; ++h) {
				for (int w = 0; w < 3; ++w) {
					int pos = h*3 + w;
					float cov = 0;
					for (int i = 0; i < data_size; ++i) {
						cov += datum.float_data(h * data_size + i) * datum.float_data(w * data_size + i);
					}
					cov/=data_size;
					// float cov = caffe_cpu_dot(data_size, &data.float_data(h*data_size), &data.float_data(w*data_size))/data_size;
					sum_blob.set_data(pos, sum_blob.data(pos) + cov);
				}
			}
		}
		++count;
		if (count % 10000 == 0) {
			LOG(INFO) << "Processed " << count << " files.";
		}
		cursor->Next();
	}

	if (count % 10000 != 0) {
		LOG(INFO) << "Processed " << count << " files.";
	}
	for (int i = 0; i < 9; ++i) {
		sum_blob.set_data(i, sum_blob.data(i)/count);
	}
	// for (int i = 0; i < sum_blob.data_size(); ++i) {
	// 	sum_blob.set_data(i, sum_blob.data(i) / count);
	// }
	// Write to disk
	if (argc == 3) {
		LOG(INFO) << "Write to " << argv[2];
		WriteProtoToBinaryFile(sum_blob, argv[2]);
	}
	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	std::vector<float> mean_values(channels, 0.0);
	LOG(INFO) << "Number of channels: " << channels;
	for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) {
			mean_values[c] += sum_blob.data(dim * c + i);
		}
		LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
	}
	return 0;
}
