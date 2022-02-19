#!/usr/bin/env bash

DATA_DIR=/home/tione/notebook/dataset

# cut the same region among frames in a video
echo "Cut videos ..."
mkdir $DATA_DIR/videos/train_without_same_region $DATA_DIR/videos/display_jpg
mkdir $DATA_DIR/videos/display_jpg/train
python ./pre/generate_videos_without_same_region.py \
-id $DATA_DIR/videos/video_5k/train_5k/ \
-od $DATA_DIR/videos/train_without_same_region/ \
-dd $DATA_DIR/videos/display_jpg/train/
echo "Finished."

# replace some videos processed badly on above step
# python ./pre/replace_bad_videos.py --train

# extract rbg and text features
echo "Extract features ..."
mkdir $DATA_DIR/processed
mkdir $DATA_DIR/processed/train_cut
python ./pre/extract_feature.py --ex_train
rm -r $DATA_DIR/processed/train_cut/.ipynb_checkpoints
echo "Finished."

# train the model
echo "Train start ..."
python train.py