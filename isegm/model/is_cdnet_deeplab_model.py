import torch.nn as nn
import torch.nn.functional as F
import torch
from isegm.utils.serialization import serialize
from .is_cdnet_model import ISModel
from .modeling.cdnet_deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead
from isegm.model.modifiers import LRMult
from isegm.model.modeling.cdnet.FDM import FDM
from isegm.model.modeling.cdnet.PDM import PDM
from isegm.model.ops import DistMapsDifferentable
import math
from segmentation_models_pytorch.losses import DiceLoss
from isegm.model.ops import DetrministicBilinear2D
from isegm.inference import utils
import cv2
import numpy as np
from copy import deepcopy
from isegm.inference.clicker import Click
from isegm.inference.transforms import AddHorizontalFlip

class DeeplabModel(ISModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
                 backbone_norm_layer=None, backbone_lr_mult=0.1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.with_prev_mask = True

        self.feature_extractor = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch, project_dropout=aspp_dropout,
                                               norm_layer=norm_layer, backbone_norm_layer=backbone_norm_layer)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))
        self.head = SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                                num_layers=2, norm_layer=norm_layer)
        self.latent_head = SepConvHead(4, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                                num_layers=2, norm_layer=norm_layer)

        self.dist_maps = DistMapsDifferentable(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        self.fdm_dist_maps = DistMapsDifferentable(norm_radius=24, spatial_scale=1.0, cpu_mode=False, use_disks=True)

        self.PDM = PDM()


    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        fdm_clicks = self.fdm_dist_maps(image, points)
        h,w = fdm_clicks.shape[-2],fdm_clicks.shape[-1]
        hs,ws = math.ceil(h/8),math.ceil(w/8)

        if torch.are_deterministic_algorithms_enabled():
            fdm_clicks = DetrministicBilinear2D(fdm_clicks, size=(hs,ws), align_corners=True)
        else:
            fdm_clicks = F.interpolate(fdm_clicks,(hs,ws),mode='bilinear',align_corners=True)

        return coord_features, fdm_clicks


    def backbone_forward(self, image, coord_features, fdm_clicks):
        backbone_features, pos_map = self.feature_extractor(image, coord_features,fdm_clicks)
        pred = self.head(backbone_features)
        latent_preds = self.latent_head(backbone_features)

        return {'instances': pred, 'fdm_instances': pos_map, 'latent_instances':latent_preds}




    def forward_optimizable(self, image_in, gt_mask, points_in, args=None, transforms=None, gt_mask_without_transforms=None):

        for MODE in ["MIN" if args.optim_min else "MAX"]:

            print("New click processing")
            image_nd = image_in.clone()

            per_step_metrics = []
            points = torch.clone(points_in)
            image = torch.clone(image_in)
            image, prev_mask = self.prepare_input(image)
            self.dist_maps.register(image, points)
            self.fdm_dist_maps.register(image, points)
            instance_loss = DiceLoss('binary', from_logits=False)

            lr = args.lr_mult * 5 * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (400 * np.sqrt(2))
            optimizer = torch.optim.Adam([self.dist_maps.last_click], lr = lr)

            best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
            best_mask_loss = 1e50
            best_params = None
            best_outputs = {}

            if self.dist_maps.last_click.requires_grad:
                iters = args.n_opt_steps
            else:
                iters = 1

            gt_8_bit = gt_mask[0][0].cpu().numpy().astype(np.uint8)

            if self.with_prev_mask:
                prev_mask_8_bit = ((0.5 * (prev_mask[0] + torch.flip(prev_mask[1], dims=[2])))[0].cpu().numpy() > 0.5).astype(np.uint8)
            else:
                prev_mask_8_bit = 1 - gt_8_bit

            gt_mask_dt_for_positive = cv2.distanceTransform(1 - (gt_8_bit - (gt_8_bit & prev_mask_8_bit)), cv2.DIST_L2, 0)
            gt_mask_dt_for_negative =  cv2.distanceTransform(gt_8_bit | (1 - prev_mask_8_bit), cv2.DIST_L2, 0)

            gt_mask_dt_for_positive = torch.from_numpy(gt_mask_dt_for_positive).to(image.device)
            gt_mask_dt_for_negative =  torch.from_numpy(gt_mask_dt_for_negative).to(image.device)

            for i in range(iters):

                optimizer.zero_grad()
                coord_features, coords_new = self.dist_maps()
                fdm_clicks, _ = self.fdm_dist_maps.forward_with_last_click(self.dist_maps.last_click)

                if image.shape[0] == 2:
                    # Flips
                    coord_features = torch.cat([coord_features, torch.flip(coord_features, (3,))])
                    fdm_clicks = torch.cat([fdm_clicks, torch.flip(fdm_clicks, (3,))])

                h,w = fdm_clicks.shape[-2],fdm_clicks.shape[-1]
                hs,ws = math.ceil(h/8),math.ceil(w/8)

                if torch.are_deterministic_algorithms_enabled():
                    fdm_clicks = DetrministicBilinear2D(fdm_clicks, size=(hs,ws), align_corners=True)
                else:
                    fdm_clicks = F.interpolate(fdm_clicks,(hs,ws),mode='bilinear',align_corners=True)

                positive_selected = (coords_new[0][0] * gt_mask_dt_for_positive)
                negative_selected = (coords_new[0][1] * gt_mask_dt_for_negative)
                regularization_loss = positive_selected.mean() + negative_selected.mean()
                curr_mask_loss = regularization_loss.item()

                if args.vis_optim:
                    new_stack =  coord_features[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    img_numpy = np.clip((image[0].cpu().permute(1, 2, 0).cpu().numpy() + 2) / 4, 0, 1) * 255
                    positive = np.dstack([new_stack[:, :, 0], new_stack[:, :, 0], new_stack[:, :, 0]])
                    negative = np.dstack([new_stack[:, :, 1], new_stack[:, :, 1], new_stack[:, :, 1]])
                    mask_stack = np.dstack([gt_mask[0][0].cpu(), gt_mask[0][0].cpu(), gt_mask[0][0].cpu()])
                    img_to_save = np.where(positive, (0, 255, 0), img_numpy).astype(np.uint8)
                    img_to_save = np.where(negative, (255, 0, 0), img_to_save).astype(np.uint8)


                if coord_features.shape[1] == 3:
                    click_map = coord_features[:,1:,:,:]
                else:
                    click_map = coord_features


                small_image = image
                small_coord_features = coord_features
                small_coord_features = self.maps_transform(small_coord_features)
                outputs = self.backbone_forward(small_image, small_coord_features, fdm_clicks)

                if torch.are_deterministic_algorithms_enabled():
                    mask = DetrministicBilinear2D(outputs['instances'], size=image.size()[2:], align_corners=True)
                else:
                    mask = nn.functional.interpolate(outputs['instances'], size=image.size()[2:], mode='bilinear', align_corners=True)

                if mask.shape[0] == 2:
                    mask, mask_f = mask[:1], mask[1:]
                    mask =  0.5 * (mask + torch.flip(mask_f, dims=[3]))
                    image_nd = image_nd[:1, :3]
                    click_map = click_map[:1]

                mask = self.PixelDiffusion(image_nd, mask, click_map)

                prediction = torch.sigmoid(mask)

                if args.vis_optim:
                    prediction_stack = np.dstack([prediction[0][0].cpu().detach(), prediction[0][0].cpu().detach(), prediction[0][0].cpu().detach()])
                    cv2.imwrite('clicks.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))

                if self.dist_maps.last_click.requires_grad:
                    main_loss = torch.mean(instance_loss(prediction, gt_mask[:1].to(prediction.device))) * (1 if MODE == 'MAX' else - 1)
                    loss = regularization_loss + (main_loss / 1000.)

                # Compute IOU in FULL resolution
                prediction = mask.detach()

                curr_params = self.dist_maps.last_click.detach().cpu().numpy()[::-1].copy()
                curr_params[0] = np.clip(curr_params[0], 0, image.shape[2] - 1)
                curr_params[1] = np.clip(curr_params[1], 0, image.shape[3] - 1)

                dummy_click = Click(is_positive=True, coords=(curr_params[0], curr_params[1]))
                for t in reversed(transforms):
                    if not isinstance(t, AddHorizontalFlip):
                        prediction, dummy_click = t.inv_transform(prediction, dummy_click, side_effects=False)

                prediction_cpu = prediction.cpu().detach().numpy()[0, 0] > args.thresh
                curr_iou = utils.get_iou(gt_mask_without_transforms, prediction_cpu)
                curr_biou = utils.get_boundary_iou(gt_mask_without_transforms, prediction_cpu)

                curr_metrics = [curr_iou, curr_biou]

                logging_record = [*curr_metrics, *dummy_click.coords, *curr_params, *list(image_in.shape), *list(image.shape)]

                if (curr_mask_loss <= best_mask_loss * 1.05) and ((curr_metrics > best_iou and MODE == 'MAX') or (curr_metrics < best_iou and MODE != 'MAX')):
                    best_mask_loss = curr_mask_loss
                    best_iou = deepcopy(curr_metrics)
                    best_params = deepcopy(curr_params)

                    print(MODE, "IoU/BIoU update: ", best_iou)

                    best_outputs = mask.detach()

                    # Flag of successful updates
                    logging_record.append(1)
                else:
                    logging_record.append(0)

                if self.dist_maps.last_click.requires_grad:
                    loss.backward()
                    optimizer.step()

                per_step_metrics.append(logging_record)

        return best_outputs, best_params, np.array(per_step_metrics)


    def PixelDiffusion(self, image, mask, clickmap):
        instance_out = self.PDM(image, mask, clickmap)
        return instance_out
