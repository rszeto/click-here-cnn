# TODOS

* Move experiments
* Add .caffemodels
* Update training curve plot code
* Add instructions for generating and training models
* Add note to install Caffe first

# Generate synthetic data

1. Set up datasets

	```
	cd datasets
	./get_pascal3d.sh
	./get_sun2012pascalformat.sh
	./get_shapenet-correspondences.sh
	```

2. Set up paths

	```
	cp global_variables.py.example global_variables.py
	### MODIFY global_variables.py ###
	python setup.py
	```

3. Compile mex code

	```
	cd render_pipeline/kde/matlab_kde_package/mex
	matlab -nodisplay -r "makemex; quit;"
	```

4. Generate viewpoint and truncation distributions with KDE

	```
	cd render_pipeline/kde
	matlab -nodisplay -r "run_sampling; quit;"
	```

5. Render, crop, and overlay backgrounds

	```
	cd render_pipeline
	python run_render.py
	python run_crop.py
	python run_overlay.py
	```


# Generate training and testing LMDBs

1. Generate the data that will eventually go into the LMDBs

	```
	cd view_estimation_correspondences
	python generate_lmdb_data.py
	```

2. Generate the LMDBs

	```
	cd view_estimation_correspondences
	python generate_lmdbs.py
	```

# Generate testing LMDBs only

This is the best option if you only want to evaluate the included models.

1. Generate the data that will eventually go into the LMDBs

	```
	cd view_estimation_correspondences
	python generate_lmdb_data.py --pascal_test_only
	```

2. Generate the LMDBs

	```
	cd view_estimation_correspondences
	python generate_lmdbs.py --pascal_test_only
	```


# Evaluating demo models

These are the models whose results are included in CH-CNN paper.

1. Replace project root and test LMDB paths in `demo_experiments`

	```
	cd view_estimation_correspondences/eval_scripts
	python init_demo_experiments.py
	```

2. Run evaluation on demo models (and cache scores to generate visualizations)

	```
	cd view_estimation_correspondences/eval_scripts
	python evaluateAcc.py 67 6000 --demo --cache_preds
	python evaluateAcc.py 70 2000 --demo --cache_preds
	```

	Results are stored in the `evaluation` folder under each experiment's folder.


# Generating visualizations

	```
	# Assumes you ran above evaluation.
	# $PROJ_ROOT is the location of the root of this project.
	cd view_estimation_correspondences/eval_scripts
	python visualize_predictions.py 6932 \
		$PROJ_ROOT/demo_experiments/000067/evaluation/cache_6000.pkl R4CNN \
		$PROJ_ROOT/demo_experiments/000070/evaluation/cache_2000.pkl CH-CNN
	```

	Results are stored under `$PROJ_ROOT/view_estimation_correspondences/eval_scripts/visualizations/qualitative_comparison`.


# Generating error distribution histograms

	```
	# Assumes you ran above evaluation.
	# $PROJ_ROOT is the location of the root of this project.
	python visualize_error_distribution.py \
		$PROJ_ROOT/demo_experiments/000067/evaluation/cache_6000.pkl R4CNN
	python visualize_error_distribution.py \
		$PROJ_ROOT/demo_experiments/000070/evaluation/cache_2000.pkl CH-CNN
	```

	Results are stored under `$PROJ_ROOT/view_estimation_correspondences/eval_scripts/visualizations/error_distribution`.
