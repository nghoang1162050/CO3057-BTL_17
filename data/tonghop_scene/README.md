# Custom scene for the Capstone (Bai tap tong hop) project

By default `src/tonghop_pipeline.py` uses scikit-image's `camera` sample
photo (the classic "Cameraman" picture) as a reproducible stand-in real
scene, so the pipeline runs with no external files.

To analyze your team's own real scene (as the assignment asks), drop a
single photo here, e.g.:

```
my_scene.jpg
```

Any `.jpg` / `.jpeg` / `.png` / `.bmp` file is accepted; if more than one
is present, the first one alphabetically is used. There is currently no
video support -- for a video, extract a representative frame (or a few
frames processed one at a time) as a still image first.

Tips for a good result with this specific pipeline:
- A scene with both some straight/rectilinear structure (walls, furniture,
  a tripod, a building) and a clear single foreground subject works best
  -- Part 3.2 (edges/lines/corners) benefits from the former, Part 3.3
  (saliency + GrabCut segmentation) from the latter.
- Avoid a subject whose color/brightness closely matches the background
  (e.g. a dark coat against dark shadow) -- GrabCut separates foreground
  from background using color, so low contrast there weakens the result
  (see the report's discussion of this exact case on the default photo).
