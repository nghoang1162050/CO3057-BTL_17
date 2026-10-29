# Computer Vision - Project 1

This project covers color channels, spatial filters, and geometric transforms. The assignment and report guidelines are in [`docs/`](docs/).

## Structure

- `src/utils.py`: sample images and figure helpers
- `src/part1_color_channels.py`: grayscale conversion and RGB channels
- `src/part2_filtering.py`: low-pass and high-pass filters
- `src/part3_transformations.py`: translation, rotation, scaling, affine and projective transforms
- `notebooks/CV_Project_1.ipynb`: walkthrough that imports the functions in `src/`
- `outputs/`: figures produced by the scripts

## Run the project

Use Python 3.9 or newer. Clone the `Project_1` branch, then run these commands from the repository root:

```bash
git clone --branch Project_1 --single-branch https://github.com/nghoang1162050/CO3057-BTL_17.git
cd CO3057-BTL_17
python -m pip install -r requirements.txt
python src/part1_color_channels.py
python src/part2_filtering.py
python src/part3_transformations.py
```

Each script prints a short result summary and saves PNG figures under its matching `outputs/part1`, `outputs/part2`, or `outputs/part3` folder. A complete run produces 14 PNG files. The inputs are sample images from `scikit-image`; no separate image download is needed.

To view and rerun the walkthrough, start Jupyter from the repository root:

```bash
jupyter notebook notebooks/CV_Project_1.ipynb
```

In Jupyter, choose **Run All Cells**. The notebook imports the code in `src/` and displays the results inline. For Google Colab, open the notebook and follow its setup cell, which clones the `Project_1` branch.

---

# Project 2 - Gradient Domain Editing and Panorama Stitching

Guidelines: [`docs/CV-project-2.pdf`](docs/CV-project-2.pdf). Notebook: `notebooks/CV_Project_2.ipynb`.

- `src/p2_gradient_domain.py`: Poisson seamless cloning solved as a sparse linear system (standard and mixed gradients), compared with a hard paste and a feathered alpha blend by a seam score.
- `src/p2_panorama.py`: SIFT and ORB features, ratio-test matching, RANSAC homography, multi-image warp and feather blending, plus OpenCV's `Stitcher` as a reference.

```bash
python src/p2_gradient_domain.py            # synthetic demo pair
python src/p2_gradient_domain.py --data data/p2_gradient_domain_teapot_wall --out outputs/p2_gradient_domain_teapot_wall
python src/p2_panorama.py                   # data/p2_panorama: our balcony photos
python src/p2_panorama.py --data data/p2_panorama_parallax --out outputs/p2_panorama_parallax
python src/p2_panorama.py --data data/p2_panorama_partial --out outputs/p2_panorama_partial
```

The gradient-domain input folders hold `background`, `source`, `mask` (`.png` or `.jpg`) and an optional `offset.txt` (`y,x`); with no complete set the synthetic pair runs. The panorama folders hold at least 3 overlapping photos. A panorama run writes about 90 MB of keypoint, match and panorama figures; only `00_inputs.png` and `04_sift_vs_orb.png` are kept in the repository.

# Capstone - Real Scene Analysis Pipeline

Guidelines: [`docs/CV-project-tonghop.pdf`](docs/CV-project-tonghop.pdf). Notebook: `notebooks/CV_Project_Tonghop.ipynb`.

- `src/tonghop_preprocess.py` (3.1): resize, CLAHE lighting normalization, bilateral denoising.
- `src/tonghop_geometry.py` (3.2): Canny edges, probabilistic Hough lines, Shi-Tomasi corners.
- `src/tonghop_content.py` (3.3): spectral-residual saliency whose bounding box seeds GrabCut.
- `src/tonghop_pipeline.py` (3.4): runs the stages and saves the before/after figures.

```bash
python src/tonghop_pipeline.py              # data/tonghop_scene/ban_cong.jpg
```

With an empty `data/tonghop_scene/`, the pipeline uses the `camera` sample from `scikit-image`.
