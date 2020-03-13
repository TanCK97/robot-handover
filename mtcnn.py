import torch
import torch.nn as nn
import time
from torchvision import transforms

from lib.headpose import module_init, head_pose_estimation
from mtcnn.mtcnn import MTCNN

from detectron2.config import get_cfg

from utils.add_config import *

class MTCNN(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.head_pose_module = module_init(cfg)
        self.mtcnn = MTCNN()
        self.transformations = transforms.Compose([transforms.Resize(224), \
                                        transforms.CenterCrop(224), transforms.ToTensor(), \
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.softmax = nn.Softmax(dim=1).cuda()

        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        """
        predictions, bounding_box, face_keypoints, w, face_area = head_pose_estimation(frame, self.mtcnn, self.head_pose_module, self.transformations, self.softmax, self.idx_tensor)

        return predictions, bounding_box

def setup_mtcnn(cfg_keypoint, confidence_threshold, weights):
    """
    ATTENTION: Must be called before mtcnn()
    """
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(cfg_keypoint)

    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.WEIGHTS = weights

    cfg.freeze()

    demo = MTCNN(cfg_object)

    setup_logger(name="MTCNN")
    logger = setup_logger()

    return cfg, demo

def mtcnn(image, cfg, demo, confidence_threshold, weights, logger):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.
        cfg (path): path to the default Keypoint RCNN config file provided by detectron2
        demo (MTCNN object): instance of the MTCNN class
        confidence_threshold (float): the confidence threshold of the network
        weights (path): path to the object detection weigths
        logger (logger):
    """
    start_time = time.time()
    predictions, bounding_box = demo.run_on_image(image)
    logger.info(
        "{}: detected {} instances in {:.2f}s".format(
            path, len(predictions["instances"]), time.time() - start_time
        )
    )

    return predictions, bounding_box
