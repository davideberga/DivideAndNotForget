import torch
from torch import nn
from .resnet_linear_turbo import resnet18, resnet50, resnet101
from networks.network import LLL_Net


class ExtractorEnsemble(LLL_Net):

    def __init__(self, backbone, taskcla, network_type, device):
        super().__init__(backbone, taskcla, remove_existing_head=False)
        self.model = None
        self.num_features = 64
        self.network_type = network_type
        if network_type == "resnet18":
            self.bb_fun = resnet18
        elif network_type == "resnet50":
            self.bb_fun = resnet50
        elif network_type == "resnet101":
            self.bb_fun = resnet101
        else:
            raise RuntimeError("Network not supported")

        self.bbs = nn.ModuleList([])
        self.head = nn.Identity()

        self.task_offset = [0]
        self.taskcla = taskcla
        self.device = device

    def add_head(self, num_outputs):
        pass

    def forward(self, x):
        features = [bb.forward(x) for bb in self.bbs]
        return torch.stack(features, dim=1)

    def freeze_backbone(self):
        pass
