import torch
import torch.nn as nn
import numpy as np
import cv2
from isegm.model.ops import DistMaps, DistMapsSAM, ScaleLayer, BatchImageNormalize, DistMapsDifferentable, DetrministicBilinear2D
from isegm.model.modifiers import LRMult
from segmentation_models_pytorch.losses import DiceLoss
from isegm.inference import utils
from isegm.inference.clicker import Click
from copy import deepcopy
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from .is_model import split_points_by_order


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ISModelSAM(nn.Module):
    def __init__(self, device='cuda', model_path=None):
        super().__init__()
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        model_type = 'vit_b' if 'vit_b' in str(model_path) else 'vit_h' if 'vit_h' in str(model_path) else 'vit_l'
        sam = sam_model_registry[model_type](checkpoint=model_path)
        for n, p in sam.named_parameters():
            p.requires_grad = False
        sam.eval()
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        self.prev_mask = None
        self.resize = ResizeLongestSide(sam.image_encoder.img_size)
        self.with_prev_mask = True
        self.binary_prev_mask = False

    def forward_optimizable(self, optimize, image_in, gt_mask, points_in, args=None, transforms=None, gt_mask_without_transforms=None):

        for MODE in ["MIN" if args.optim_min else "MAX"]:

            print("New click processing")

            if points_in.shape[1] == 2:
                self.prev_mask = None

            per_step_metrics = []
            points = torch.clone(points_in).cpu().numpy()
            image, prev_mask = self.prepare_input(image_in)

            input_label = []
            points_list = []
            input_image = self.resize.apply_image_torch(image * 255)

            gt_mask_resized = self.resize.apply_image_torch(gt_mask)
            gt_mask_resized = (gt_mask_resized > 0.5).float()

            self.sam_predictor.set_torch_image(input_image, image.shape[2:])

            for idx in range(points.shape[1]):
                if points[0][idx][-1] < 0:
                    continue
                if idx >= points.shape[1] // 2:
                    # Negative click
                    points_list.append(np.hstack([0, points[0][idx]]))
                else:
                    points_list.append(np.hstack([1, points[0][idx]]))

            points_list = np.array(points_list)
            all_list = points_list[points_list[:, 3].argsort()]
            points_list = all_list[:, 1:3][:, ::-1].copy()
            input_label = list(all_list[:, 0])

            old_h, old_w = image.shape[2:]
            new_h, new_w = get_preprocess_shape(old_h, old_w, self.sam_predictor.transform.target_length)

            points_list = points_list.astype(float)
            points_list[..., 0] = points_list[..., 0] * (new_w / old_w)
            points_list[..., 1] = points_list[..., 1] * (new_h / old_h)
            points_list = torch.as_tensor(points_list, dtype=torch.float, device=self.sam_predictor.device)[None]
            input_label = torch.tensor(input_label)[None].to(self.sam_predictor.device)

            self.dist_maps.register(input_image)

            last_point = torch.nn.Parameter(points_list[:, -1:, :])

            last_label = input_label[:, -1:]
            other_points = points_list[:, :-1, :]
            other_labels = input_label[:, :-1]

            instance_loss = DiceLoss('binary', from_logits=False)

            lr = args.lr_mult * 5 * np.sqrt(image.shape[-1] ** 2 + image.shape[-2] ** 2) / (400 * np.sqrt(2))
            optimizer = torch.optim.Adam([last_point], lr = lr)

            best_iou = [-2, -2] if MODE == 'MAX' else [2, 2]
            best_mask_loss = 1e50
            best_params = None
            best_prev_mask = None
            best_outputs = {}

            if last_point.requires_grad:
                iters = args.n_opt_steps
            else:
                iters = 1

            gt_8_bit = gt_mask[0][0].cpu().numpy().astype(np.uint8)
            prev_mask_8_bit = (prev_mask[0].cpu().numpy() > 0.5)[0].astype(np.uint8)

            gt_mask_dt_for_positive = cv2.distanceTransform(1 - (gt_8_bit - (gt_8_bit & prev_mask_8_bit)), cv2.DIST_L2, 0)
            gt_mask_dt_for_negative =  cv2.distanceTransform(gt_8_bit | (1 - prev_mask_8_bit), cv2.DIST_L2, 0)

            gt_mask_dt_for_positive = torch.from_numpy(gt_mask_dt_for_positive).to(image.device)
            gt_mask_dt_for_negative =  torch.from_numpy(gt_mask_dt_for_negative).to(image.device)


            for i in range(iters):

                optimizer.zero_grad()
                coords_new = self.dist_maps(last_point, last_label)
                gt_mask_dt_for_positive_resized = nn.functional.interpolate(gt_mask_dt_for_positive[None, None], size=coords_new.shape[2:], mode='bilinear', align_corners=True)
                gt_mask_dt_for_negative_resized = nn.functional.interpolate(gt_mask_dt_for_negative[None, None], size=coords_new.shape[2:], mode='bilinear', align_corners=True)

                # Two distance transforms for interaction location loss
                positive_selected = (coords_new[0][0] * gt_mask_dt_for_positive_resized[0][0])
                negative_selected = (coords_new[0][1] * gt_mask_dt_for_negative_resized[0][0])
                regularization_loss = positive_selected.mean() + negative_selected.mean()
                curr_mask_loss = regularization_loss.item()

                if args.vis_optim:
                    new_stack =  coords_new[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                    img_numpy = np.clip(input_image[0].cpu().permute(1, 2, 0).cpu().numpy(), 0, 255)
                    positive = np.dstack([new_stack[:, :, 0], new_stack[:, :, 0], new_stack[:, :, 0]])
                    negative = np.dstack([new_stack[:, :, 1], new_stack[:, :, 1], new_stack[:, :, 1]])
                    mask_stack = np.dstack([gt_mask_resized[0][0].cpu(), gt_mask_resized[0][0].cpu(), gt_mask_resized[0][0].cpu()])
                    img_to_save = np.where(positive, (0, 255, 0), img_numpy).astype(np.uint8)
                    img_to_save = np.where(negative, (255, 0, 0), img_to_save).astype(np.uint8)

                res, scores, logits = self.sam_predictor.predict_torch(
                            point_coords=torch.cat([other_points, last_point], dim=1),
                            point_labels=torch.cat([other_labels, last_label], dim=1),
                            mask_input=self.prev_mask,
                            multimask_output=True,
                            return_logits=True)

                prediction = torch.sigmoid(res)
                prediction = prediction[0, torch.argmax(scores)][None, None]

                if args.vis_optim:
                    prediction_stack = nn.functional.interpolate(prediction.detach(), mode='bilinear', align_corners=True, size=mask_stack.shape[:2])
                    prediction_stack = np.dstack([prediction_stack[0][0].cpu(), prediction_stack[0][0].cpu(), prediction_stack[0][0].cpu()])
                    cv2.imwrite('clicks.png', np.hstack([cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR), 255 * mask_stack, 255 * prediction_stack]))

                if last_point.requires_grad:
                    main_loss = torch.mean(instance_loss(prediction, gt_mask.to(prediction.device).contiguous())) * (1 if MODE == 'MAX' else - 1)
                    loss = regularization_loss + (main_loss / 1000.)

                curr_params = last_point[0][0].detach().cpu().numpy()
                curr_params[0] *= (old_w / new_w)
                curr_params[1] *= (old_h / new_h)
                curr_params = curr_params[::-1]
                curr_params[0] = np.clip(curr_params[0], 0, image.shape[2] - 1)
                curr_params[1] = np.clip(curr_params[1], 0, image.shape[3] - 1)

                # Compute IOU in FULL resolution
                prediction_detached = prediction.detach()
                prediction_cpu = prediction_detached.cpu().detach().numpy()[0, 0] > args.thresh
                curr_iou = utils.get_iou(gt_mask_without_transforms, prediction_cpu)
                curr_biou = utils.get_boundary_iou(gt_mask_without_transforms, prediction_cpu)

                curr_metrics = [curr_iou, curr_biou]
                logging_record = [*curr_metrics, *last_point.detach().cpu().numpy(), *curr_params, *list(image_in.shape), *list(image.shape)]

                if (curr_mask_loss <= best_mask_loss * 1.05) and ((curr_metrics > best_iou and MODE == 'MAX') or (curr_metrics < best_iou and MODE != 'MAX')):
                    best_mask_loss = curr_mask_loss
                    best_iou = deepcopy(curr_metrics)
                    best_params = deepcopy(curr_params)

                    print(MODE, "IoU/BIoU update: ", best_iou)

                    best_outputs = {'instances': prediction.detach()}
                    best_prev_mask = logits[0, torch.argmax(scores)][None, None].detach()

                    # Flag of successful updates
                    logging_record.append(1)
                else:
                    logging_record.append(0)

                if last_point.requires_grad:
                    loss.backward()
                    optimizer.step()

            per_step_metrics.append(logging_record)

        # Since SAM use its own prev mask format
        self.prev_mask = best_prev_mask

        return best_outputs, best_params, np.array(per_step_metrics)


    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()
        return image, prev_mask


    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features
