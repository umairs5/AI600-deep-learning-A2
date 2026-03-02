import torch
import torch.nn as nn
import numpy as np

class ChampionMLP(nn.Module):
    def __init__(self, input_size=784, num_classes=15):
        super(ChampionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChampionMLP().to(device)
model.load_state_dict(torch.load('champion_model.pth', map_location=device))
model.eval()

test_data = np.load('quickdraw_test.npz')
X_test = test_data['test_images'].astype('float32') / 255.0
X_test_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)

preds = predictions.cpu().numpy()
pred_string = ','.join(map(str, preds))
print(pred_string)
