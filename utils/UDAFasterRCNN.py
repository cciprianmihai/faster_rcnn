import torch
import torch.nn as nn


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None  # Return None for alpha gradient

class DomainClassifier(nn.Module):
    def __init__(self, in_features, classes_no=3):
        super(DomainClassifier, self).__init__()
        self.domain_fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, classes_no)
        )

    def forward(self, x):
        return self.domain_fc(x)



class UDAFasterRCNN(nn.Module):
    def __init__(self, model, domain_classes_no=3):
        super().__init__()
        self.model = model
        self.domain_classifier_p = DomainClassifier(256*8*8*2, domain_classes_no)
        self.domain_classifier_loss = nn.CrossEntropyLoss()
        self.grl = GradientReversalLayer.apply

    def forward(self, images, targets=None, domain=0, alpha=1.0):
        if self.training:
            loss_dict = self.model(images, targets)
            det_loss = sum(loss for loss in loss_dict.values())

            images_transformed, _ = self.model.transform(images, targets)
            feats = self.model.backbone(images_transformed.tensors)

            pooled = []
            for lvl in ["0", "1"]:
                feat = feats[lvl]
                p    = nn.AdaptiveAvgPool2d((8,8))(feat)
                pooled.append(p.flatten(1))
            p = torch.cat(pooled, dim=1)

            p_rev = self.grl(p, alpha)
            logits = self.domain_classifier_p(p_rev)             # still raw scores
            # probs  = torch.sigmoid(logits)                       # map to [0,1]

            batch_size = p.size(0)
            labels = torch.full(
                (batch_size,),
                domain,
                dtype=torch.long,
                device=logits.device
            )

            domain_loss = self.domain_classifier_loss(logits, labels)

            return det_loss, domain_loss, logits, labels

        else:
            return self.model(images)
