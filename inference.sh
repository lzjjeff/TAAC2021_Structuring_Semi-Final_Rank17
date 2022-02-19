#!/usr/bin/env bash

DATA_DIR=/home/tione/notebook/dataset

# cut the same region among frames in a video
echo "Cut videos ..."
mkdir $DATA_DIR/videos/test_2nd_without_same_region $DATA_DIR/videos/display_jpg
mkdir $DATA_DIR/videos/display_jpg/test_2nd
python ./pre/generate_videos_without_same_region.py \
-id $DATA_DIR/videos/test_5k_2nd/ \
-od $DATA_DIR/videos/test_2nd_without_same_region/ \
-dd $DATA_DIR/videos/display_jpg/test_2nd/
echo "Finished."

# replace some videos processed badly on above step
# python ./pre/replace_bad_videos.py --inference

# extract rbg and text features
echo "Extract features ..."
mkdir $DATA_DIR/processed/test_2nd_cut
python ./pre/extract_feature.py --ex_test
rm -r $DATA_DIR/processed/test_2nd_cut/.ipynb_checkpoints

python inference.py
python ./src/utils/postProcess.py --file_path ./save/ --file_name results.json