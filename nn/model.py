import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor

class DigitModel(nn.Module):
    def __init__(self, num_chars) -> None:
        super(DigitModel, self).__init__()
        # self.convnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def ctc_loss(self, x, targets):
        batch_size = x.size(1)
        log_softmax = nn.functional.log_softmax(x, 2)
        input_length = torch.full(size=(batch_size, ), fill_value=log_softmax.size(0), dtype=torch.int32)
        target_length = torch.full(size=(batch_size, ), fill_value=targets.size(1), dtype=torch.int32)

        loss = nn.CTCLoss(blank=0)(log_softmax, targets, input_length, target_length)

        return loss


    def forward(self, images, targets=None):
        batch_size, _, _, _ = images.size()
        x = self.conv_layers(images)
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, x.size(1), -1)
        x = self.linear_layers(x)
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)
        if targets is not None:
            loss = self.ctc_loss(x, targets)
            return x, loss

        return x, None
        
if __name__ == "__main__":
    model = DigitModel(10)
    img = torch.rand(1, 1, 75, 300)
    target = torch.randint(1, 20, (1, 5))
    x, loss = model(img, target)
    print(x.size())
