import torch
import numpy as np
from learning.modules.cuda_module import CudaModule
from logger import Logger
from visualization import Presenter
from parameters.parameter_server import get_current_parameters
class GoalPredictionGoodCriterion(CudaModule):
    """
    This module takes a given goal-prediction mask and a correct goal location mask and
    finds whether the argmax location in the predicted mask is close to the argmax location in the ground truth mask
    On the trajectory-prediction + control model (CoRL), this is used in the following way:
        If the goal prediction is good, we train the controller to execute the trajectory
        If the goal prediction is bad, we skip the gradient update
    """
    def __init__(self, ok_distance=3.2, logger=None):
        self.logger = logger
        super(GoalPredictionGoodCriterion, self).__init__()
        self.ok_distance = ok_distance

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        return self

    def forward(self, masks, mask_labels, show="", iteration=0):

        if show != "":
            Presenter().show_image(masks.data[0], "pred_mask", torch=True, waitkey=1, scale=4)
            Presenter().show_image(mask_labels.data[0], "mask_labels", torch=True, waitkey=1, scale=4)
            self.logger.log_image("pred_mask", Presenter().prep_image(masks.data[0], 4), iteration) 
            self.logger.log_image("mask_labels", Presenter().prep_image(mask_labels.data[0], 4), iteration) 
        if masks.size(1) == 1:
            return False

        # TODO: Handle batches if necessary
        goal_mask = masks[0, 1, :, :]
        goal_mask_flat = goal_mask.view([1, -1])
        max_val, argmax = goal_mask_flat.max(1)
        argmax_loc_x = argmax / goal_mask.size(1)
        argmax_loc_y = torch.remainder(argmax, goal_mask.size(1))
        argmax_loc = torch.cat([argmax_loc_x.unsqueeze(1), argmax_loc_y.unsqueeze(1)], 1)

        goal_mask_l = mask_labels[0, 1, :, :]
        goal_mask_flat_l = goal_mask_l.view([1, -1])
        max_val, argmax_l = goal_mask_flat_l.max(1)
        argmax_loc_x_l = argmax_l / goal_mask_l.size(1)
        argmax_loc_y_l = torch.remainder(argmax_l, goal_mask_l.size(1))
        argmax_loc_l = torch.cat([argmax_loc_x_l.unsqueeze(1), argmax_loc_y_l.unsqueeze(1)], 1)

        dist = (argmax_loc - argmax_loc_l).float().norm(dim=1)
        success = dist < self.ok_distance

        return success
