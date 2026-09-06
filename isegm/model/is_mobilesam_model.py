import torch
from isegm.model.ops import DistMapsSAM
from mobile_segment_anything import SamPredictor
from mobile_segment_anything.utils.transforms import ResizeLongestSide
from .is_sam_model import ISModelSAM
from mobile_encoder.setup_mobile_sam import setup_model


class ISModelMobileSAM(ISModelSAM):
    def __init__(self, device='cuda', model_path=None):
        super().__init__()
        self.dist_maps = DistMapsSAM(norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True)
        checkpoint = torch.load(model_path)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        for n, p in mobile_sam.named_parameters():
            p.requires_grad = False
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.sam_predictor = SamPredictor(mobile_sam)
        self.prev_mask = None
        self.resize = ResizeLongestSide(mobile_sam.image_encoder.img_size)
        self.with_prev_mask = True
        self.binary_prev_mask = False
