# Computer Vision - Project 1

This project covers color channels, spatial filters, and geometric transforms. The assignment and report guidelines are in [`docs/`](docs/).

## Structure

- `src/utils.py`: sample images and figure helpers
- `src/part1_color_channels.py`: grayscale conversion and RGB channels
- `src/part2_filtering.py`: low-pass and high-pass filters
- `src/part3_transformations.py`: translation, rotation, scaling, affine and projective transforms
- `notebooks/CV_Project_1.ipynb`: walkthrough that imports the functions in `src/`
- `outputs/`: figures produced by the scripts

## Run

Use Python 3.9 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/part1_color_channels.py
python src/part2_filtering.py
python src/part3_transformations.py
jupyter notebook notebooks/CV_Project_1.ipynb
```

The scripts save figures under `outputs/part1`, `outputs/part2`, and `outputs/part3`. The notebook displays results inline. Both use sample images from `scikit-image`; no separate image download is needed.
