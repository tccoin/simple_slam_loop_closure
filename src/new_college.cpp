#include <DBoW2/DBoW2.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "frame_descriptor.h"
#include "utils.h"

int main(int argc, char *argv[])
{
  std::string vocabulary_path("/src/data/surf64_k10L6.voc.gz");
  slc::FrameDescriptor descriptor(vocabulary_path);
  int interval = std::stoi(argv[1]);

  std::string dataset_folder("/kitti/image_2/");
  std::vector<std::string> filenames;

  for (int i = 0; i < 2761; i += interval)
  {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << i;
    std::string file_name = ss.str() + ".png";
    filenames.push_back(file_name);
  }

  std::cerr << "Processing " << filenames.size() << " images\n";

  // Will hold BoW representations for each frame
  std::vector<DBoW2::BowVector> bow_vecs;
  bow_vecs.reserve(filenames.size());

  for (unsigned int img_i = 0; img_i < filenames.size(); img_i++)
  {
    auto img_filename = dataset_folder + filenames[img_i];
    auto img = cv::imread(img_filename);

    std::cout << img_filename << "\n";

    if (img.empty())
    {
      std::cerr << std::endl
                << "Failed to load: " << img_filename << std::endl;
      exit(1);
    }

    // Get a BoW description of the current image
    DBoW2::BowVector bow_vec;
    descriptor.describe_frame(img, bow_vec);
    bow_vecs.push_back(bow_vec);
  }

  std::cerr << "\nWriting output...\n";

  std::string output_path("results/confusion_" + std::to_string(interval) +
                          ".txt");
  std::ofstream of;
  of.open(output_path);
  if (of.fail())
  {
    std::cerr << "Failed to open output file " << output_path << std::endl;
    exit(1);
  }

  // Compute confusion matrix
  // i.e. the (i, j) element of the matrix contains the distance
  // between the BoW representation of frames i and j
  for (int i = 0; i < bow_vecs.size(); i++)
  {
    for (int j = i + 1; j < bow_vecs.size(); j++)
    {
      of << descriptor.vocab_->score(bow_vecs[i], bow_vecs[j]) << " ";
    }
    of << "\n";
  }

  of.close();
  std::cerr << "Output done\n";
}
