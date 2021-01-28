#!/bin/bash
# Transforms Bernhard's data into usable format

# SOURCE_TRAIN_DIR=data_small/train
# SOURCE_VAL_DIR=data_small/val
# 
# DEST_TRAIN_DIR=reformatted_data_small/train
# DEST_VAL_DIR=reformatted_data_small/val

SOURCE_TRAIN_DIR=data/train
SOURCE_VAL_DIR=data/val

DEST_TRAIN_DIR=reformatted_data/train
DEST_VAL_DIR=reformatted_data/val


mkdir -p $DEST_TRAIN_DIR/img
mkdir -p $DEST_TRAIN_DIR/normals
mkdir -p $DEST_VAL_DIR/img
mkdir -p $DEST_VAL_DIR/normals

for i in $SOURCE_TRAIN_DIR/*/*/img_*; do cp "$i" $DEST_TRAIN_DIR/img/; done
for i in $SOURCE_TRAIN_DIR/*/*/normals_*; do cp "$i" $DEST_TRAIN_DIR/normals/; done
for i in $SOURCE_VAL_DIR/*/*/img_*; do cp "$i" $DEST_VAL_DIR/img/; done
for i in $SOURCE_VAL_DIR/*/*/normals_*; do cp "$i" $DEST_VAL_DIR/normals/; done

ls $DEST_TRAIN_DIR/img | cut -c5- | rev | cut -c5- | rev > $DEST_TRAIN_DIR/names.txt
ls $DEST_VAL_DIR/img | cut -c5- | rev | cut -c5- | rev > $DEST_VAL_DIR/names.txt

mkdir $DEST_TRAIN_DIR/downsampled_normals
mkdir $DEST_VAL_DIR/downsampled_normals

mkdir $DEST_TRAIN_DIR/model_output

mkdir $DEST_TRAIN_DIR/downsampled_normals
mkdir $DEST_VAL_DIR/downsampled_normals

mkdir $DEST_TRAIN_DIR/model_output
mkdir $DEST_VAL_DIR/model_output
