import torch
import time

from detectron2.config import get_cfg
from utils.add_config import *
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine.defaults import DefaultPredictor

class FasterRCNN(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata_object = MetadataCatalog.get(
            "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
        """
        predictions = self.predictor(image)

        return predictions

def setup_fcnn(cfg_object, confidence_threshold, weights):
    """
    ATTENTION: Must be called before faster_rcnn()
    """

    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(cfg_object)

    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = weights

    cfg.freeze()

    demo = FasterRCNN(cfg_object)

    setup_logger(name="Faster RCNN")
    logger = setup_logger()

    return cfg, demo

def faster_rcnn(image, cfg, demo, confidence_threshold, weights, logger):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.
        cfg (path): path to the default Faster RCNN config file provided by detectron2
        demo (FasterRCNN object): instance of the FasterRCNN class
        confidence_threshold (float): the confidence threshold of the network
        weights (path): path to the object detection weigths
        logger (logger):
    """
    start_time = time.time()
    predictions = demo.run_on_image(image)
    logger.info(
        "{}: detected {} instances in {:.2f}s".format(
            path, len(predictions["instances"]), time.time() - start_time
        )
    )

    return predictions
