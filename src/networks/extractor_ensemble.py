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

        # Uncomment to load a model, set 6 to number of experts that's in .pth, comment backbone training
        # self.bbs = nn.ModuleList([copy.deepcopy(bb) for _ in range(min(len(taskcla), 6))])
        # for bb in self.bbs:
        #     bb.fc = nn.Identity()
        # state_dict = torch.load("seb-resnet32.pth")
        # self.load_state_dict(state_dict, strict=True)

        self.task_offset = [0]
        self.taskcla = taskcla
        self.device = device

    def add_head(self, num_outputs):
        pass

    def forward(self, x):
        # semi_features = self.bbs[0].calculate_semi_features(x)
        features = [bb.forward(x) for bb in self.bbs]
        return torch.stack(features, dim=1)

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        pass
