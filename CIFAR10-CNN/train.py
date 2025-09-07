import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb
import numpy as np
from tqdm import tqdm
import time
import os

hyperparameter_defaults = dict(
    dropout=0.25,
    batch_size=128,

    learning_rate=0.001,
    epochs=2,

    base_channels=32,
    channel_mult=2,
    n_conv_layers=10,

    apply_dropout_at=2,

    activation='ReLU',

    # Leave these for now
    kernel_size=3, stride=2, padding=1,
)

# Init WandB
wandb.login(key="7aa9946530b11dba10807ceb8e1f50e2d89f3824")
wandb.init(config=hyperparameter_defaults)
config = wandb.config

ACTIVATIONS = {
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
    'SiLU': nn.SiLU(),
    'LeakyReLU': nn.LeakyReLU(0.01)
}
activation_func = ACTIVATIONS[config.activation]

class CNN(nn.Module):
    def __init__(self, shape_info):
        super(CNN, self).__init__()

        self.base_channels = config.base_channels
        self.channel_mult = config.channel_mult
        self.n_conv_layers = config.n_conv_layers
        self.apply_dropout_at = config.apply_dropout_at

        kernel_size = config.kernel_size
        stride = config.stride
        padding = config.padding

        # Build conv layers dynamically
        self.conv_layers = nn.ModuleList()
        in_channels = shape_info['num_channels']
        for i in range(self.n_conv_layers):
            out_channels = self.base_channels * (self.channel_mult ** i)
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            in_channels = out_channels

        self.activation = activation_func
        self.dropout1 = nn.Dropout(config.dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # put near end: yields one value per channel

        # Apply dropout to last set layers

        # Final channels after all conv layers
        final_channels = self.base_channels * (self.channel_mult ** (self.n_conv_layers - 1))
        self.fc = nn.Linear(final_channels, shape_info['num_classes'])

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = self.activation(conv_layer(x))

            # fun idea to try this.
            if self.apply_dropout_at is not None and i >= self.apply_dropout_at:
                x = self.dropout1(x)

        x = self.global_avg_pool(x)  # one value per channel
        return self.fc(x.flatten(start_dim=1))

def initialize_dataset():
    info = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info['mean'], info['std'])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)
    sample_img, _ = train_dataset[0]
    shape_info = {
        'num_channels': sample_img.shape[0],
        'img_height': sample_img.shape[1],
        'img_width': sample_img.shape[2],
        'num_classes': len(train_dataset.classes),
        'class_names': train_dataset.classes
    }

    return train_dataset, test_dataset, train_loader, test_loader, shape_info


def train():
    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Initialize Dataset
    train_dataset, test_dataset, train_loader, test_loader, shape_info = initialize_dataset()

    # Set Model
    model = CNN(shape_info).to(device)

    # Set loss stuff
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(config.epochs):
        # Set train vars
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train += target.size(0)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct_train / total_train:.2f}%'
            })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train

        # Evaluation phase
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_test += pred.eq(target.view_as(pred)).sum().item()
                total_test += target.size(0)

        test_acc = 100. * correct_test / total_test

        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        })

        print(
            f'Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    wandb.finish()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


if __name__ == "__main__":
    # Set Seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the train
    start_time = time.time()
    train()
    training_time = time.time() - start_time
    print(f"CNN Training completed in {training_time:.2f} seconds")