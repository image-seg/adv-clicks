import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide, ResizeTrans
from isegm.inference.clicker import Click
import numpy as np
from copy import deepcopy

class DiffisionPredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 infer_size = 384,
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

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.crop_l = infer_size
        self.transforms.append(ResizeTrans(384))
        self.transforms.append(SigmoidForPred())
        self.focus_roi = None
        self.global_roi = None
        self.with_flip = True
        if self.with_flip:
            self.transforms.append(AddHorizontalFlip())


    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()

    def get_prediction(self, clicker, prev_mask=None, args=None):
        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y,last_x = click.coords[0],click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

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

        prediction, last_click, metrics_dict = self._get_prediction(image_nd, clicks_lists, is_image_changed, gt_mask, args, self.transforms, clicker.gt_mask)
        #print(1,prediction.max(), prediction.min()) #这里变成binary了

        click_removed = clicker._remove_last_click()
        last_click = Click(is_positive=click_removed.is_positive, coords=(last_click[0], last_click[1]))

        for t in reversed(self.transforms):
            if not isinstance(t,AddHorizontalFlip):
                prediction, last_click = t.inv_transform(prediction, last_click)
        #print(2,prediction.max(), prediction.min())
        clicker.add_click(last_click)

        prediction = prediction > 0.5
        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0], metrics_dict


    def _get_prediction(self, image_nd, clicks_lists, is_image_changed, gt_mask=None, args=None, transforms=None, gt_mask_without_transforms=None):
        points_nd = self.get_points_nd(clicks_lists)
        model_outputs = self.net.forward_optimizable(image_nd, gt_mask, points_nd, args, transforms, gt_mask_without_transforms)
        return model_outputs[0], model_outputs[1], model_outputs[2]


    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)
        print('_set_transform_states')

    def apply_transforms(self, image_nd, gt_mask, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            prev_clicks = deepcopy(clicks_lists)
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            gt_mask, _ = t.transform(gt_mask, prev_clicks)
            is_image_changed |= t.image_changed
        gt_mask = (gt_mask > 0.5).float()
        return image_nd, gt_mask, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_points_nd_inbbox(self, clicks_list, y1,y2,x1,x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg =len(clicks_list) - num_pos
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [],[]
        for click in clicks_list:
            flag,y,x,index = click.is_positive, click.coords[0],click.coords[1], 0
            y,x = map_point_in_bbox(y,x,y1,y2,x1,x2,self.crop_l)
            if flag:
                pos_clicks.append( (y,x,index))
            else:
                neg_clicks.append( (y,x,index) )

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)




    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        print('set')
