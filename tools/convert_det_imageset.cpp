// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <sstream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
		"When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
		"Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
				"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
		"When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
		"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
		"Optional: What type should we encode the image as ('png','jpg',...).");

typedef struct Annos {
	int ID;
	int pos[4];
} Anno;

void loadmap(std::map<std::string, int> &map) {
	fstream fsmap;
	fsmap.open("/ceph/sun/DB/ILSVRC2015b/map_det.txt", std::fstream::in);
	std::string line;
	
	while (std::getline(fsmap, line)) {
		std::istringstream iss(line);
		std::string key;
		int value;
		if (!(iss >> key >> value)) { break;}
		map.insert(std::pair<std::string, int>(key,value));
	}
	fsmap.close();
	//LOG(INFO) << "Map_det done.";
}

void loadXML(std::string xml, std::vector<Anno> &vAnnos, std::map<std::string, int> &map) {
	fstream fs;
	fs.open(xml, std::fstream::in);
	std::string line;

	Anno tempAnno;

	//LOG(INFO) << xml << " under processing.";

	while (std::getline(fs, line)) {
		size_t foundObject = line.find("<object>",0);
		if (foundObject!=std::string::npos) {
			std::getline(fs, line);
			size_t foundName = line.find("<name>",0);
			std::string temp;
			temp.assign(line,foundName+6,9);

			tempAnno.ID = map[temp];

			std::getline(fs, line);
			while (line.find("<bndbox>",0) == std::string::npos) {
				std::getline(fs, line);
			}
			for (int i = 0; i < 4; ++i)	{
				std::getline(fs, line);
				size_t foundA = line.find(">",0);
				size_t foundB = line.find("<",foundA);
				std::string ttemp;
				ttemp.assign(line,foundA+1,foundB-foundA-1);
				tempAnno.pos[i] = std::stoi(ttemp,nullptr);
			}

			vAnnos.push_back(tempAnno);
		}
	}
	fs.close();
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
				"format used as input for Caffe.\n"
				"Usage:\n"
				"    convert_imageset [FLAGS] LISTFILE \n"
				"The ImageNet dataset for the training demo is at\n"
				"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 1) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
		return 1;
	}

	const bool is_color = !FLAGS_gray;
	// const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;

	std::ifstream infile(argv[1]);
	std::vector<std::pair<std::string, int> > lines;
	std::string filename;
	int label;
	while (infile >> filename >> label) {
		lines.push_back(std::make_pair(filename, label));
	}
	if (FLAGS_shuffle) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		shuffle(lines.begin(), lines.end());
	}
	LOG(INFO) << "A total of " << lines.size() << " images.";

	if (encode_type.size() && !encoded)
		LOG(INFO) << "encode_type specified, assuming encoded=true.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	scoped_ptr<db::DB> db_image(db::GetDB(FLAGS_backend));
	scoped_ptr<db::DB> db_anno(db::GetDB(FLAGS_backend));
	db_image->Open("DET_train", db::NEW);
	db_anno->Open("DET_train_anno", db::NEW);
	scoped_ptr<db::Transaction> txn_image(db_image->NewTransaction());
	scoped_ptr<db::Transaction> txn_anno(db_anno->NewTransaction());

	// Storing to db
	std::string image_root_folder("/ceph/sun/DB/ILSVRC2015b/Data/DET/train/");
	std::string anno_root_folder("/ceph/sun/DB/ILSVRC2015b/Annotations/DET/train/");
	Datum image_datum;
	Datum anno_datum;

	// Create map
	std::map<std::string, int> map_det;
	loadmap(map_det);

	int count = 0;
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	// int data_size = 0;
	// bool data_size_initialized = false;

	for (int line_id = 0; line_id < lines.size(); ++line_id) {
		bool status;
		std::string enc = encode_type;
		if (encoded && !enc.size()) {
			// Guess the encoding type from the file name
			string fn = lines[line_id].first;
			size_t p = fn.rfind('.');
			if ( p == fn.npos )
				LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
			enc = fn.substr(p);
			std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
		}
		status = ReadImageToDatum(image_root_folder + lines[line_id].first + ".JPEG",
				lines[line_id].second, resize_height, resize_width, is_color,
				enc, &image_datum);
		if (status == false) continue;

		// const std::string& data = image_datum.data();
		// CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
		// 		<< data.size();
		// // }
		
		// sequential
		int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
				lines[line_id].first.c_str());

		// Put in db
		string out_image;
		CHECK(image_datum.SerializeToString(&out_image));
		txn_image->Put(string(key_cstr, length), out_image);

		// Do anno
		std::vector<Anno> vAnnos;
		std::string xml = anno_root_folder + lines[line_id].first + ".xml";
		loadXML(xml,vAnnos,map_det);

		// Blob<float> anno_blob;

		if (vAnnos.size() != 0) {
			// anno_blob.Reshape(1,vAnnos.size(),1,5);

			// float* anno_blob_data = anno_blob.mutable_cpu_data();
			// int tempPos = 0;


			anno_datum.set_height(1);
			anno_datum.set_width(5);
			anno_datum.set_channels(vAnnos.size());
			anno_datum.clear_data();
			anno_datum.clear_float_data();

			for (std::vector<Anno>::iterator it = vAnnos.begin(); it != vAnnos.end(); ++it)	{
				anno_datum.add_float_data(static_cast<float>(it->ID));
				for (int i = 0; i < 4; ++i) {
					anno_datum.add_float_data(static_cast<float>(it->pos[i]));
				}
			}

			// for (int d = 0; d < dim_features; ++d) {
			// 	anno_datum.add_float_data(anno_blob_data[d]);
			// }

		}
		else {
			for (int d = 0; d < 5; ++d) {
				anno_datum.add_float_data(0);
			}
		}

		string out_anno;
		CHECK(anno_datum.SerializeToString(&out_anno));
		txn_anno->Put(string(key_cstr, length), out_anno);

		if (++count % 1000 == 0) {
			// Commit db
			txn_image->Commit();
			txn_image.reset(db_image->NewTransaction());
			txn_anno->Commit();
			txn_anno.reset(db_anno->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn_image->Commit();
		txn_anno->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}
	return 0;
}
