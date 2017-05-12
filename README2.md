# Click Here: Human-Localized Keypoints as Guidance for Viewpoint Estimation
Ryan Szeto and Jason J. Corso, University of Michigan

## Using this code

If you use this work in your research, please cite the following paper:

	@article{szeto2017click,
	  title={Click Here: Human-Localized Keypoints as Guidance for Viewpoint Estimation},
	  author={Szeto, Ryan and Corso, Jason J},
	  journal={arXiv preprint arXiv:1703.09859},
	  year={2017}
	}

I ran this code from scratch, so it should work. However, feel free to contact me at `szetor [at] umich [dot] edu` if you have trouble.

## Introduction

This code implements the work described in the arXiv report ["Click Here: Human-Localized Keypoints as Guidance for Viewpoint Estimation"](https://arxiv.org/abs/1703.09859). It extends the [Render for CNN project](https://github.com/shapenet/RenderForCNN) by generating semantic keypoint data alongside rendered (or real) image data. This code lets you generate the training and testing data that we used in our paper's experiments, as well as reproduce the numbers presented in the paper.

Please note that this code is insanely inefficient storage-wise, since it stores dense keypoint maps and keypoint class vectors on disk as LMDBs. Excluding code and model weights, you only need about 20 GB to run our pre-trained models on the PASCAL 3D+ test set, but you need at least XX TB for our entire training process over synthetic and real image examples. We are implementing a more efficient version of this code in TensorFlow, which might be made available someday...

## Reproducing our results

In this project, we have included the pre-trained models used to produce the results in the paper. This section outlines how to run them.

### Set up dataset

First, install the PASCAL 3D+ dataset as follows. The Bash scripts in this readme assume you are starting from this project's root directory unless otherwise noted.

	cd datasets
	./get_pascal3d.sh

### Set up weights

The weights to our models are included in an external file available [here](https://umich.box.com/shared/static/ukikg3hogr0azsn2o1dheyq4jmj07qj1.gz). Download this folder to the project root, and then extract the contents to the `demo_experiments` folder:

	tar -xzvf ch-cnn-model-weights.tar.gz -C demo_experiments

### Set up Caffe

Our code uses a customized version of Caffe. It is based on Caffe RC1, and the main difference is that it includes the custom layers from the Caffe version used by Render for CNN (their Caffe source code is available [here](https://github.com/charlesq34/caffe-render-for-cnn)).

TODO

### Set up global variables

Copy the example global variables file, edit the paths as instructed, and propagate the variables to the demo experiment setups.

	cp global_variables.py.example global_variables.py
	### MODIFY global_variables.py ###
	python setup.py
	python init_demo_experiments.py

### Generate PASCAL 3D+ LMDBs

Generate the test instance data, then generate the test LMDBs.

	cd view_estimation_correspondences
	python generate_lmdb_data.py --pascal_test_only
	python generate_lmdbs.py --pascal_test_only

### Run experiments

Run the evaluation code on our demo models (located in `demo_experiments`). It takes about an hour on a NVIDIA GeForce GTX 980 Ti GPU for each experiment, so I recommend running this overnight and/or on a cluster environment, if possible.

	cd view_estimation_correspondences/eval_scripts
	python evaluateAcc.py 67 6000 --demo --cache_preds
	python evaluateAcc.py 68 0 --demo --cache_preds
	python evaluateAcc.py 70 2000 --demo --cache_preds
	python evaluateAcc.py 71 0 --demo --cache_preds
	python evaluateAcc.py 72 0 --demo --cache_preds
	python evaluateAcc.py 73 0 --demo --cache_preds
	python evaluateAcc.py 78 0 --demo --cache_preds
	python evaluateAcc.py 80 0 --demo --cache_preds
	python evaluateAcc.py 81 0 --demo --cache_preds
	python evaluateAcc.py 82 0 --demo --cache_preds
	python evaluateAcc.py 83 0 --demo --cache_preds
	python evaluateAcc.py 84 0 --demo --cache_preds
	python evaluateAcc.py 85 0 --demo --cache_preds
	python evaluateAcc.py 86 0 --demo --cache_preds
	python evaluateAcc.py 87 0 --demo --cache_preds
	python evaluateAcc.py 89 2000 --demo --cache_preds
	python evaluateAcc.py 90 2000 --demo --cache_preds
	python evaluateAcc.py 93 2000 --demo --cache_preds
	python evaluateAcc.py 94 2000 --demo --cache_preds
	python evaluateAcc.py 98 4400 --demo --cache_preds
	python evaluateAcc.py 99 2000 --demo --cache_preds
	python evaluateAcc.py 100 2000 --demo --cache_preds

Results are stored in `demo_experiments/<exp_num>/evaluation`.

### Generate visualizations

The above commands cache the scores for each rotation angle, which can be compared with the `visualize_predictions.py` script. The commands below compare the predictions of fine-tuned Render for CNN and our CH-CNN model.

	# $PROJ_ROOT is the location of the root of this project.
	cd view_estimation_correspondences/eval_scripts
	python visualize_predictions.py 6932 \
		$PROJ_ROOT/demo_experiments/000067/evaluation/cache_6000.pkl R4CNN \
		$PROJ_ROOT/demo_experiments/000070/evaluation/cache_2000.pkl CH-CNN

Results are stored under `$PROJ_ROOT/view_estimation_correspondences/eval_scripts/visualizations/qualitative_comparison`.

### Generate error distribution plots

The distribution of errors for a model can be visualized with the `visualize_error_distribution.py` script. The commands below generate this plot for fine-tuned Render for CNN and our CH-CNN model.

	# $PROJ_ROOT is the location of the root of this project.
	python visualize_error_distribution.py \
		$PROJ_ROOT/demo_experiments/000067/evaluation/cache_6000.pkl R4CNN
	python visualize_error_distribution.py \
		$PROJ_ROOT/demo_experiments/000070/evaluation/cache_2000.pkl CH-CNN

Results are stored under `$PROJ_ROOT/view_estimation_correspondences/eval_scripts/visualizations/error_distribution`.

## Generating training data

This section describes how to generate synthetic and real image training data with our code. Before you execute the steps below, make sure you have set up the PASCAL 3D+ dataset, Caffe and global variables as described in ["Reproducing our results"](#reproducing-our-results).

### Set up datasets

You will need to download some auxiliary data and save the `.zip` files in the `datasets` folder. First, you need to download the following synsets from [ShapeNet](https://www.shapenet.org/): 02924116 (buses), 02958343 (cars), and 03790512 (motorcycles). Then, download our ShapeNet keypoints dataset [here](http://web.eecs.umich.edu/~jjcorso/extdelivery/shapenet-keypoints-1.0.zip). Finally, run the extraction scripts below:

	cd datasets
	./get_sun2012pascalformat.sh
	./get_shapenet-correspondences.sh

### Render synthetic images and keypoint information

#### Compile mex code

	cd render_pipeline/kde/matlab_kde_package/mex
	matlab -nodisplay -r "makemex; quit;"

#### Generate viewpoint and truncation distributions with KDE

	cd render_pipeline/kde
	matlab -nodisplay -r "run_sampling; quit;"

#### Render, crop, and overlay backgrounds

This takes many days on multiple cores. See the `global_variables.py.example` file for tips on how to make this as fast as possible.

	cd render_pipeline
	python run_render.py
	python run_crop.py
	python run_overlay.py

#### Generate training and testing LMDBs

This takes at least a day on multiple cores.

	cd view_estimation_correspondences
	python generate_lmdb_data.py
	python generate_lmdbs.py

## Training our models

TODO

### Monitoring training progress

TODO

<!--
# TODOS

* Update training curve plot code
* Add instructions for generating and training models
* Add caffe-r4cnn as submodule
* Add note to install Caffe first
-->
