import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeClassifier(nn.Module):

    def __init__(
        self,
        feat_dim=256,
        num_classes=7,
        scale=20.0
    ):
        super().__init__()

        self.scale = scale
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, feat_dim)
        )

        nn.init.xavier_normal_(
            self.prototypes
        )

    def forward(
        self,
        feat,
    ):
        """
        feat:
        [B,C,H,W]
        """

        feat = F.normalize(
            feat,
            p=2,
            dim=1,
        )
        proto = F.normalize(
            self.prototypes,
            p=2,
            dim=1,
        )

        logits = torch.einsum(
            "bchw,nc->bnhw",
            feat,
            proto,
        )

        return logits * self.scale
    def consistency_loss(
        self,
        feat,
        target,
    ):
    
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=feat.shape[2:],
            mode="nearest",
        ).squeeze(1).long()
    
        feat = F.normalize(
            feat,
            p=2,
            dim=1,
        )
    
        proto = F.normalize(
            self.prototypes,
            p=2,
            dim=1,
        )
    
        feat = feat.permute(
            0,
            2,
            3,
            1,
        )
    
        valid = target != 255
    
        feat = feat[valid]
    
        target = target[valid]
    
        target_proto = proto[target]
    
        cos = F.cosine_similarity(
            feat,
            target_proto,
            dim=1,
        )
    
        loss = (1 - cos).mean()
    
        return loss
    # def orthogonal_loss(self):
    
    #     proto = F.normalize(
    #         self.prototypes,
    #         p=2,
    #         dim=1
    #     )
    
    #     gram = torch.matmul(
    #         proto,
    #         proto.t()
    #     )
    
    #     eye = torch.eye(
    #         gram.size(0),
    #         device=gram.device
    #     )
    
    #     loss = ((gram - eye) ** 2).mean()
    
    #     return loss


class SceneAwarePrototypeClassifier(nn.Module):

    def __init__(
        self,
        feat_dim=256,
        num_classes=7,
        num_scenes = 2,
        scale=20.0
    ):
        super().__init__()

        self.scale = scale
        self.prototypes = nn.Parameter(
                torch.randn(
                    num_scenes,
                    num_classes,
                    feat_dim
                )
        )
        nn.init.xavier_normal_(
            self.prototypes
        )

    def forward(
        self,
        feat,
        scene_lable
    ):
        """
        feat:
        [B,C,H,W]
        """

        feat = F.normalize(
            feat,
            p=2,
            dim=1,
        )
        proto = F.normalize(
            self.prototypes[scene_lable],
            p=2,
            dim=2,
        )

        logits = torch.einsum(
            "bchw,bnc->bnhw",
            feat,
            proto,
        )

        return logits * self.scale