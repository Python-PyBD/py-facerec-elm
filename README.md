# py-facerec-elm

Face recognition using PCA (Principal Component Analysis) and ELM (Extreme Learning Machine).

# How to start

Download and install [HPELM](https://github.com/akusok/hpelm), `python-scipy`, `python-numpy` and `python-scikit`.

# Dataset

`Cambridge_FaceDB` is used for this project.

# Files

|Filename|Purpose|
|---|---|
|`extract_features.py`|Reads all images in `Cambridge_FaceDB` folder, applies PCA compression and saves image data into `input.txt` and class data into `output.txt`|
|`hidden_node_count.py`|Finds out correct number of hidden nodes for error free recognition.|
|`facerec_elm.py`|The main program that tests images and prints their ID|
