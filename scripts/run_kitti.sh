echo -e "exec /src/build/kitti_loop_detection /src/data/surf64_k10L6.voc.gz /kitti/00/image_2/ /src/results/confusion_seq00_5.txt 5\nexec /src/build/kitti_loop_detection /src/data/surf64_k10L6.voc.gz /kitti/02/image_2/ /src/results/confusion_seq02_5.txt 5\nexec /src/build/kitti_loop_detection /src/data/surf64_k10L6.voc.gz /kitti/05/image_2/ /src/results/confusion_seq05_5.txt 5\nexec /src/build/kitti_loop_detection /src/data/surf64_k10L6.voc.gz /kitti/06/image_2/ /src/results/confusion_seq06_5.txt 5\nexec /src/build/kitti_loop_detection /src/data/surf64_k10L6.voc.gz /kitti/07/image_2/ /src/results/confusion_seq07_5.txt 5\nexec /src/build/kitti_loop_detection /src/data/surf64_k10L6.voc.gz /kitti/08/image_2/ /src/results/confusion_seq08_5.txt 5" | parallel