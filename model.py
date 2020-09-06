import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class Model:
    def __init__(self,checkpoint_url="COCO-Detection/retinanet_R_101_FPN_3x.yaml",threshold=0.7):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(checkpoint_url))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
        self.predictor = DefaultPredictor(cfg)
    
    def _get_filtered_output(self,required_classes, old_outputs, seg=False):
        old_output = old_outputs["instances"]
        indices = np.isin(old_output.pred_classes.cpu().numpy(), required_classes)
        new_output = detectron2.structures.instances.Instances(image_size=old_output.image_size)
        new_output.pred_boxes = old_output.pred_boxes[indices]
        new_output.scores = old_output.scores[indices]
        new_output.pred_classes = old_output.pred_classes[indices]
        if seg:
            new_output.pred_masks = old_output.pred_masks[indices]
        return {'instances': new_output}
    
    def get_class_outputs(self,im, required_classes=[0]):
        all_outputs = []

        outputs = self.predictor(im)
        outputs = self._get_filtered_output(required_classes, outputs)
        all_outputs.append(outputs)
        return all_outputs