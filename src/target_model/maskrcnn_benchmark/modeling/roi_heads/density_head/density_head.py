import torch
import torch.nn as nn


def _make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class DensityHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.backend_feat = [256, 256, 256, 128, 64]
        self.down_sample = nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, dilation=2, padding=2)
        self.backend = _make_layers(self.backend_feat, in_channels=256, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.density_criterion = nn.MSELoss(reduction='sum')
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError('not support')

    def forward(self, x, detections, targets=None):
        x = self.down_sample(x)
        x = self.backend(x)
        x = self.output_layer(x)
        predication_density_map = x

        for i, detection in enumerate(detections):
            detection.add_field('density_map', predication_density_map[i].squeeze())

        if not self.training:
            return x, detections, {}

        batch_size = x.shape[0]
        gt_density_maps = torch.stack([target.get_field('density_map').density_map for target in targets], dim=0).unsqueeze(1)
        density_loss = self.density_criterion(input=predication_density_map, target=gt_density_maps) / (batch_size * 2)
        return x, detections, dict(density_loss=density_loss)


def build_roi_density_head(in_channels):
    return DensityHead(in_channels=in_channels)
