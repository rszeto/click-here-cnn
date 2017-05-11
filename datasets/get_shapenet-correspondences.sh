#!/bin/bash
# This script unzips the ShapeNet models and keypoint dataset into the proper directory
# for rendering. Before running, check that you have the following zip files in this
# directory:
#  From ShapeNet website:
#      - 02924116.zip
#      - 02958343.zip
#      - 03790512.zip
#  From Ryan Szeto's or Jason Corso's website:
#      - shapenet-keypoints-1.0.zip


dataset_dir="shapenet-correspondences"
synset_ids=("02924116" "02958343" "03790512")
shapenet_keypoints_file="shapenet-keypoints-1.0.zip"

# Check ShapeNet synset zips exist
for synset_id in "${synset_ids[@]}"; do
	if ! [ -e "$synset_id.zip" ]; then
		echo "Could not find $synset_id.zip. See script for details."
		exit 1
	fi
done

# Check keypoints zip exists
if ! [ -e "$shapenet_keypoints_file" ]; then
	echo "Could not find $shapenet_keypoints_file. See script for details."
	exit 1
fi

echo "Unzipping..."

mkdir "$dataset_dir"
unzip "$shapenet_keypoints_file" -d "$dataset_dir"
for synset_id in "${synset_ids[@]}"; do
	unzip "$synset_id.zip" -d "$dataset_dir"
done

echo "Done."
