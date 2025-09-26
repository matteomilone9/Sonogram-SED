import warnings
from torch import nn, optim

from data_loader import *
from trainer import trainer_
from aed_models import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

### === PARAMS === ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
h = 128
learning_rate = 1e-3
epochs = 30
batch_size = 512
batch_size_2 = 64

ae_model = NS_1().to(device)
optimizer = optim.AdamW(ae_model.parameters(), lr=1e-3,  weight_decay=1e-5)
criterion = nn.MSELoss()


### === UNSUPERVISED === ###
ae_dataset = AudioDataset(
    audio_dir="audio_anomali/dataset_1s",
    supervised=False)

# Split unsupervised
train_size = int(0.75 * len(ae_dataset))
val_size = len(ae_dataset) - train_size 
ae_train, ae_val = random_split(ae_dataset, [train_size, val_size])

ae_train_loader = DataLoader(ae_train, batch_size=batch_size, shuffle=True)
ae_val_loader = DataLoader(ae_val, batch_size=batch_size, shuffle=False)

### === STARTER === ###
trainer_(model=ae_model, train_loader=ae_train_loader, val_loader=ae_val_loader,
         criterion=criterion, optimizer=optimizer, device=device, num_epochs=epochs,
         results_dir="V6_results_1s", csv_name="training_log_1s_V6.csv", model_name="best_model_1s_V6.pth")