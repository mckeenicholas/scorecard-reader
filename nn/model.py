import torch
from torch import nn, Tensor
from torch.nn import functional as F

class CRNN(nn.Module):
    def __init__(self, num_chars, num_classes) -> None:
        super(CRNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(3, 1)),
            nn.ReLU(),
        )

        # BiLSTM layers
        self.lstm = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=0.2),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=0.2)
        )

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)


    def forward(self, images, targets=None):
        x = self.conv_layers(images)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.lstm(x)
        x = self.fc(x)

        return x

        
if __name__ == "__main__":
    model = CRNN(10)
    img = torch.rand(1, 1, 75, 300)
    target = torch.randint(1, 20, (1, 5))
    x, loss = model(img, target)
    print(x.size())
