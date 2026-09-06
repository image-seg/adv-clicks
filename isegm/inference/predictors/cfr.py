import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.inference.clicker import Click
from .base import BasePredictor

class CFRPredictor(BasePredictor):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 cascade_step=0,
                 cascade_adaptive=False,
                 cascade_clicks=1,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.cascade_step = cascade_step
        self.cascade_adaptive = cascade_adaptive
        self.cascade_clicks = cascade_clicks

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())


    def get_prediction(self, clicker, prev_mask=None, on_cascade=False, cascade_iter=0, args=None, optimize=True):
        clicks_list = clicker.get_clicks()

        if len(clicks_list) <= self.cascade_clicks and self.cascade_step > 0 and not on_cascade:
            for i in range(self.cascade_step):
                prediction, self.metrics_dict = self.get_prediction(clicker, None, True, i, args, optimize)
                if self.cascade_adaptive and prev_mask is not None:
                    diff_num = (
                        (prediction > 0.49) != (prev_mask > 0.49)
                    ).sum()
                    if diff_num <= 20:
                        return prediction, self.metrics_dict
                prev_mask = prediction
            return prediction, self.metrics_dict

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, gt_mask, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, torch.from_numpy(clicker.gt_mask[None, None, ...]).float(), [clicks_list]
        )

        pred_logits, last_click, self.metrics_dict = self._get_prediction((cascade_iter == 0) and optimize, image_nd, clicks_lists, is_image_changed, gt_mask, args, self.transforms, clicker.gt_mask)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                size=image_nd.size()[2:])
        click_removed = clicker._remove_last_click()
        last_click = Click(is_positive=click_removed.is_positive, coords=(last_click[0], last_click[1]))
        for t in reversed(self.transforms):
            prediction, last_click = t.inv_transform(prediction, last_click)
        clicker.add_click(last_click)


        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker, prev_mask, on_cascade, cascade_iter, args, optimize=False)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0], self.metrics_dict
