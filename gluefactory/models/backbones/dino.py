import torch
import torch.nn.functional as F

from ..base_model import BaseModel


class Dino(BaseModel):
    default_conf = {"weights": "dino_vits8", "allow_resize": False}
    required_data_keys = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load("facebookresearch/dino:main", conf.weights)
        self.set_initialized()

    def _forward(self, data):
        img = data["image"]
        import pdb; pdb.set_trace()
        if self.conf.allow_resize:
            img = F.upsample(img, [int(x // 14 * 14) for x in img.shape[-2:]])
        desc, cls_token = self.net.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]

        return {
            "features": desc,
            "global_descriptor": cls_token,
            "descriptors": desc.flatten(-2).transpose(-2, -1),
        }

    def loss(self, pred, data):
        raise NotImplementedError
