import torch.nn as nn

class clssimp(nn.Module):
    def __init__(self, num_classes, ch=2880):
        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1920, bias=True),
            nn.BatchNorm1d(1920),
            nn.ReLU(inplace=True),
        )

        self.way2 = nn.Sequential(
            nn.Linear(1920, 960, bias=True),
            nn.BatchNorm1d(960),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(960, num_classes, bias=True)

    def forward(self, x):
        # bp()
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        x = self.way2(x)
        logits = self.cls(x)
        return logits