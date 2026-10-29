# Computer Vision - CO3057 Assignments

This repository holds three assignments. The task descriptions and report guidelines are in [`docs/`](docs/).

| Assignment | Topic | Code | Notebook |
|---|---|---|---|
| [Project 1](#project-1---color-channels-filtering-and-geometric-transforms) | Color channels, spatial filters, geometric transforms | `src/part*.py` | `notebooks/CV_Project_1.ipynb` |
| [Project 2](#project-2---gradient-domain-editing-and-panorama-stitching) | Poisson blending, panorama stitching (SIFT vs ORB) | `src/p2_*.py` | `notebooks/CV_Project_2.ipynb` |
| [Capstone](#capstone---real-scene-analysis-pipeline) | Preprocessing, geometry and content analysis of one real scene | `src/tonghop_*.py` | `notebooks/CV_Project_Tonghop.ipynb` |

## Setup

Tested with Python 3.13. From a terminal:

```bash
git clone https://github.com/nghoang1162050/CO3057-BTL_17.git
cd CO3057-BTL_17
python -m pip install -r requirements.txt
```

All commands below run from the repository root. Each script prints a short summary and saves PNG figures under `outputs/`. The notebooks import the code in `src/` and show the results inline; start them with `jupyter notebook notebooks/<name>.ipynb` and choose **Run All Cells**.

For Google Colab, run `!git clone https://github.com/nghoang1162050/CO3057-BTL_17.git`, then `%cd CO3057-BTL_17` and `!pip install -r requirements.txt`, then open a notebook from the `notebooks/` folder.

## Structure

```
src/        code (utils.py is shared by all three assignments)
notebooks/  one walkthrough per assignment, saved with outputs
data/       our own photos used by Project 2 and the capstone
outputs/    figures produced by the scripts
docs/       assignment descriptions and the report guidelines
```

---

## Project 1 - Color Channels, Filtering and Geometric Transforms

- `src/part1_color_channels.py`: grayscale conversion and RGB channels
- `src/part2_filtering.py`: low-pass and high-pass filters
- `src/part3_transformations.py`: translation, rotation, scaling, affine and projective transforms

```bash
python src/part1_color_channels.py
python src/part2_filtering.py
python src/part3_transformations.py
```

A complete run saves 14 PNG files under `outputs/part1`, `outputs/part2` and `outputs/part3`. The inputs are sample images from `scikit-image`; no image download is needed.

---

## Project 2 - Gradient Domain Editing and Panorama Stitching

Guidelines: [`docs/CV-project-2.pdf`](docs/CV-project-2.pdf).

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

---

## Capstone - Real Scene Analysis Pipeline

Guidelines: [`docs/CV-project-tonghop.pdf`](docs/CV-project-tonghop.pdf).

- `src/tonghop_preprocess.py` (3.1): resize, CLAHE lighting normalization, bilateral denoising.
- `src/tonghop_geometry.py` (3.2): Canny edges, probabilistic Hough lines, Shi-Tomasi corners.
- `src/tonghop_content.py` (3.3): spectral-residual saliency whose bounding box seeds GrabCut.
- `src/tonghop_pipeline.py` (3.4): runs the stages and saves the before/after figures.

```bash
python src/tonghop_pipeline.py              # data/tonghop_scene/ban_cong.jpg
```

With an empty `data/tonghop_scene/`, the pipeline uses the `camera` sample from `scikit-image`.
