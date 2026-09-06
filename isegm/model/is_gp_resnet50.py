import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_gp_model import ISGPModel
from isegm.model.ops import ScaleLayer
from .modeling.deeplab_v3_gp import DeepLabV3Plus
from isegm.model.modifiers import LRMult
from segmentation_models_pytorch.losses import DiceLoss
from isegm.model.ops import DetrministicBilinear2D
from isegm.inference import utils
import cv2
import numpy as np
from isegm.inference.clicker import Click
from copy import deepcopy

class GpModel(ISGPModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.,
                 backbone_norm_layer=None, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, weight_dir=None, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.model = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch,
                                   project_dropout=aspp_dropout, norm_layer=norm_layer,
                                   backbone_norm_layer=backbone_norm_layer, weight_dir=weight_dir)

        side_feature_ch = 256

        self.model.apply(LRMult(backbone_lr_mult))


        mt_layers = [
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=16, out_channels=side_feature_ch, kernel_size=3, stride=1, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
        self.maps_transform = nn.Sequential(*mt_layers)
        self.L=256
        self.feature_dim = 48
        self.theta = nn.Linear(self.feature_dim+3,self.L)
        omega = 0.25*torch.randn(self.L,1)
        self.omega = nn.Parameter(omega, requires_grad=True)
        omega_var = torch.tensor(0.025)
        self.omega_var = nn.Parameter(omega_var, requires_grad=True)

        logsigma2 = torch.ones(self.feature_dim)
        self.logsigma2 = nn.Parameter(logsigma2, requires_grad=True)
        self.u_mlp = nn.Sequential(
            nn.Linear(self.feature_dim+3,96),
            nn.ReLU(True),
            nn.Linear(96,1)
        )

        weight = torch.zeros(1)
        self.weights = nn.Parameter(weight, requires_grad=True)
        self.eps2 = 1e-2

    def set_status(self, training):
        if training:
            self.eps2=1e-2
        else:
            self.eps2=1e-7



    def forward_optimizable(self, image_in, gt_mask, points_in, args=None, transforms=None, gt_mask_without_transforms=None):

        for MODE in ["MIN" if args.optim_min else "MAX"]:

            print("New click processing")
            per_step_metrics = []
            points = torch.clone(points_in)
            image = torch.clone(image_in)
            image, prev_mask = self.prepare_input(image)
            self.dist_maps.register(image, points)
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
            prev_mask_8_bit = (prev_mask[0][0].cpu().numpy() > 0.5).astype(np.uint8)

            gt_mask_dt_for_positive = cv2.distanceTransform(1 - (gt_8_bit - (gt_8_bit & prev_mask_8_bit)), cv2.DIST_L2, 0)
            gt_mask_dt_for_negative =  cv2.distanceTransform(gt_8_bit | (1 - prev_mask_8_bit), cv2.DIST_L2, 0)

            gt_mask_dt_for_positive = torch.from_numpy(gt_mask_dt_for_positive).to(image.device)
            gt_mask_dt_for_negative =  torch.from_numpy(gt_mask_dt_for_negative).to(image.device)

            for i in range(iters):

                optimizer.zero_grad()
                coord_features, coords_new = self.dist_maps()

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

                if image.shape[0] == 2:
                    coord_features = torch.cat([coord_features, torch.flip(coord_features, (3,))])

                coord_features = torch.cat((prev_mask, coord_features), dim=1)

                coord_features = self.maps_transform(coord_features)

                feature = self.model(image, coord_features)
                feature = F.normalize(feature, dim=1)

                if torch.are_deterministic_algorithms_enabled():
                    feature = DetrministicBilinear2D(feature, size=image.size()[2:], align_corners=True)
                else:
                    feature = nn.functional.interpolate(feature, size=image.size()[2:], mode='bilinear', align_corners=True)

                feature = torch.cat([feature, image], 1)

                points[:, self.dist_maps.last_click_idx] = torch.tensor([[torch.clamp(self.dist_maps.last_click[1], 0, image.shape[2] - 1),
                                                                          torch.clamp(self.dist_maps.last_click[0], 0, image.shape[3] - 1),
                                                                          self.dist_maps.last_number]],
                                                                          device=self.dist_maps.last_click.device,
                                                                          dtype=torch.float64)

                pss, label_list = self.prepare_points_labels(points, feature)

                if self.training:
                    omega = self.omega+self.omega_var.clamp(min=0.01,max=0.05)*torch.randn(self.L,1).to(feature.device)
                else:
                    omega = self.omega

                prior = self.Pathwise_GP_prior(feature, omega)
                out, u_loss = self.Pathwise_GP_update(points, feature,pss,label_list,prior,omega)
                outputs = {'instances': out}

                prediction = outputs['instances']
                prediction = torch.sigmoid(prediction)

                if args.vis_optim:
                    prediction_stack = np.dstack([prediction[0][0].cpu().detach(), prediction[0][0].cpu().detach(), prediction[0][0].cpu().detach()])
                    cv2.imwrite('clicks.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))

                if self.dist_maps.last_click.requires_grad:
                    main_loss = torch.mean(instance_loss(prediction, gt_mask.to(prediction.device))) * (1 if MODE == 'MAX' else - 1)
                    loss = regularization_loss + (main_loss / 1000.)

                # Compute IOU in FULL resolution
                prediction = nn.functional.interpolate(outputs['instances'].detach(), mode='bilinear', align_corners=True, size=image_in.size()[2:])

                curr_params = self.dist_maps.last_click.detach().cpu().numpy()[::-1].copy()
                curr_params[0] = np.clip(curr_params[0], 0, image.shape[2] - 1)
                curr_params[1] = np.clip(curr_params[1], 0, image.shape[3] - 1)

                dummy_click = Click(is_positive=True, coords=(curr_params[0], curr_params[1]))
                for t in reversed(transforms):
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

                    best_outputs = {}
                    for key in outputs.keys():
                        if outputs[key] is not None:
                            best_outputs[key] = outputs[key].detach()

                    # Flag of successful updates
                    logging_record.append(1)
                else:
                    logging_record.append(0)

                if self.dist_maps.last_click.requires_grad:
                    loss.backward()
                    optimizer.step()

                per_step_metrics.append(logging_record)

        return best_outputs, best_params, np.array(per_step_metrics)
