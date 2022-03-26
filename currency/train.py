# https://towardsdatascience.com/how-to-train-detectron2-on-custom-object-detection-data-be9d1c233e4

# git clone https://github.com/facebookresearch/detectron2.git
# python -m pip install -e detectron2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog