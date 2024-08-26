# Image-Colorization
This Python code uses OpenCV and a pre-trained Caffe model to colorize a black-and-white or grayscale image. The process involves converting the image to the Lab color space, applying a deep learning model to predict the color channels, and then combining these predictions with the original image to produce a colorized version.

## Installation
git clone https://github.com/jlsabi/image-colorization.git
cd image-colorization

### Prerequisites

Ensure you have Python 3.6+ installed on your system. You also need to install the following Python packages:

```bash
pip install opencv-python numpy
