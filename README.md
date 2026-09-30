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
