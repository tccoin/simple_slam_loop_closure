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
#include <experimental/filesystem>

#include "frame_descriptor.h"
#include "utils.h"

int main(int argc, char *argv[])
{
  std::string vocabulary_path(argv[1]);
  std::string dataset_folder(argv[2]);
  std::string output_path(argv[3]);
  int interval = std::stoi(argv[4]);

  int fileCount = 0;
  std::vector<std::string> filenames;
  for (const auto &entry : std::experimental::filesystem::directory_iterator(dataset_folder))
  {
    if (std::experimental::filesystem::is_regular_file(entry.status()))
    {
      fileCount++;
      filenames.push_back(entry.path().filename().string());
    }
  }

  // Sort filenames
  std::sort(filenames.begin(), filenames.end());

  // print dataset information
  std::cout << "Number of frames: " << fileCount << std::endl;
  std::cout << "First 10 filenames:"<<std::endl;
  for (int i = 0; i < 10; i++)
  {
    std::cout << filenames[i] << std::endl;
  }

  // Will hold BoW representations for each frame
  std::vector<DBoW2::BowVector> bow_vecs;
  bow_vecs.reserve(filenames.size());

  // Load vocabulary
  slc::FrameDescriptor descriptor(vocabulary_path);

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

  std::ofstream of;
  of.open(output_path);
  if (of.fail())
  {
    std::cerr << "Failed to open output file " << output_path << std::endl;
    exit(1);
  }

  // Compute confusion matrix (only lower triangular part)
  // i.e. the (i, j) element of the matrix contains the distance
  // between the BoW representation of frames i and j

  // First row should be all 0s
  for (int j = 0; j < bow_vecs.size() - 1; j++)
  {
    of << "0,";
  }
  of << "0\n";
  // Remaining rows (j>=i should be 0)
  for (int i = 1; i < bow_vecs.size(); i++)
  {
    for (int j = 0; j < bow_vecs.size(); j++)
    {
      if (j < i)
      {
        of << descriptor.vocab_->score(bow_vecs[i], bow_vecs[j]) << ",";
      }
      else if (j < bow_vecs.size() - 1)
      {
        of << "0,";
      }
      else
      {
        of << "0";
      }
    }
    if (i < bow_vecs.size() - 1)
    {
      of << "\n";
    }
  }

  of.close();
  std::cerr << "Output done\n";
}
