# Visual SLAM with YOLOv7 for Human Detection

This repository contains a script for tracking camera orientation in the video.

To do this, it calculates homography 
matrices for all pairs of adjacent video frames. Homography matrices are calculated using feature points filtered 
by human boxes detected by YOLOv7.

## Quick Start

<details>
<summary>Installation</summary>

Step 0. Install [Python](https://www.python.org/downloads/) and [PyTorch](https://pytorch.org/get-started/locally/#start-locally).  
Developed and tested with Python 3.9.7 and PyTorch 1.12.1+cu116.

Step 1. Clone the repository locally.
```shell
git clone https://github.com/uvd174/Visual-SLAM-YOLOv7.git --recurse-submodules
cd Visual-SLAM-YOLOv7
```

Step 2. Install dependencies.
```shell
pip install -r requirements.txt
```


</details>

<details>
<summary>Usage</summary>

```shell
python process.py --input <path_to_video> --output_dir <path_to_output_dir> --device <device>
```

* path_to_video - path to the video file in mp4 format;  
* path_to_output_dir - path to the directory where the output files will be saved;  
* device - device to run the script on. Can be `cpu` or device id e.g. `0`, `1`, etc.

</details>