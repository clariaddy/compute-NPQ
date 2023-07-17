# Computing Npq from 2D Images

This Python script performs various image processing operations on a set of images to compute NPQ (Non-Photochemical Quenching). The script is named `compute_npq.py` and requires Python 3 or later.

## Dependencies

- Git: Install Git and use `git clone` to get the repository, including examples in subfolders.
- Python3 or later with pip
- Python libraries(see `requirements.txt`)

## Usage

1. Install Git on your system. 
2. Install Python 3.x.
3. Check if pip is installed for your target Python version: `python3.x -m pip --version`. If pip is not found, install it using the following commands:

`wget https://bootstrap.pypa.io/get-pip.py`

`python3.x get-pip.py`

`rm -f get-pip.py`

4. Open a terminal or command prompt and clone the repository:

`git clone https://gitlab.com/clariaddy/compute_npq.git`

5. Install Python dependencies:

`cd compute_npq`

`python3 -m pip install --upgrade -r requirements.txt`

6. Install an appropriate text editor to edit the code, if needed.

7. To compute NPQ, add your files in the input data folder (`data/`) and modify the file names in the "Image Loading and Processing" loop accordingly. This is the only task you need to perform. The script loads the image using `PIL.Image.open()` and converts it to a NumPy array of type `float32`. Various properties of the image, such as the approximate center coordinates and radius, are then calculated. The intensity is computed by taking the mean of the pixel values along the radial direction.

## Sample Input Files

A sample of input `.png` files should be available in the `data` folder and can be used as described previously.

## Author and acknowledgment
The authors would like to acknowledge Jean-Baptiste Keck for providing valuable feedback on the code.

## Citation
If you use this script, please cite it as below:
#APA
Uwizeye, C.(2023). Compute-NPQ (Version 1.0.0) [Python script]. https://doi.org/10.5281/zenodo.8155231

#BibTex
@software{Uwizeye_compute_NPQ_2023,
  author = {Uwizeye, Clarisse},
  doi = {10.5281/zenodo.8155231},
  month = {07},
  title = {{Compute-NPQ}},
  url = {https://github.com/clariaddy/compute-NPQ},
  version = {1.0.0},
  year = {2023}
}

## License
This project is licensed under the MIT License.


