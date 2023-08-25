# Simple loop closure for Visual SLAM

[![CircleCI](https://circleci.com/gh/nicolov/simple_slam_loop_closure.svg?style=shield)](https://circleci.com/gh/nicolov/simple_slam_loop_closure)

<img src="https://github.com/nicolov/simple_slam_loop_closure/raw/master/confusion_matrix_example.png" width="500" style="text-align: center"/>

Possibily the simplest example of loop closure for Visual SLAM. More
information [on my blog](http://nicolovaligi.com/bag-of-words-loop-closure-visual-slam.html).

As I'm experimenting with alternative approaches for SLAM loop closure, I
wanted a baseline that was reasonably close to state-of-the art approaches.
The approach here is pretty similar to ORB-SLAM's, and uses SURF descriptors
and *bag of words* to translate them to a global image description vector.

## The dataset

The vocabulary file is availble at [here](https://github.com/tccoin/simple_slam_loop_closure/releases/download/v0.1/surf64_k10L6.voc.gz).

## Building with Docker

You can build and run the code using `docker-compose` and Docker. The Docker
configuration uses a Ubuntu 16.04 base image, and builds OpenCV 3 from source.

```
# Download the data files
./scripts/download_data.sh

# Will take ~10 minutes to download and build OpenCV 3
docker-compose build runner
# Enter the docker shell
docker-compose run runner bash
# You're now in a shell inside the Docker container, build and run the code:
./scripts/build.sh
./build/kitti_loop_detection ./data/brief_k10L6.voc.gz ./data/kitti ./results/output.txt 5
```

## Compatibility

Only tested on Ubuntu 16.04 LTS with OpenCV3, gcc 5.4.0

## Plotting the confusion matrix

The `ground_truth_comparison.py` plots and compares the loop closures from the
ground truth to the actual results from the code.
