import numpy as np
import torch.optim as optim
from test import compute_result

from models import *
from utils import *

# Load the configuration
config = load_config('config.json')

# Access to parameters
num_inputs = config['num_inputs']
resolution = config['resolution']
input_max = config['input_max']
input_min = config['input_min']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
epoch_size = config['epoch_size']


inputs = build_inputs(num_inputs, resolution, input_max, input_min)

input_scaler = MinMaxScaler()
inputs = input_scaler.fit_transform(inputs)
inputs = torch.tensor(inputs, dtype=torch.float32)
train_dataset = IndexedTensorDataset(inputs)

train_loader = torch.utils.data.DataLoader(train_dataset.tensors, batch_size=batch_size, shuffle=False)

model = SimpleNN()
model_loss = LossNet()

criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer_loss = optim.SGD(model_loss.parameters(), lr=0.1)

losses = []
losses_pred_loss = []
loss_scores_glob = [1]

losses_vector = torch.ones(len(train_dataset.tensors))
predicted_losses_vector = torch.ones(len(train_dataset.tensors))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Initialize lists to store training and test errors
test_errors = []

for epoch in range(num_epochs):
    batch_num = 0
    model.train()
    model_loss.train()

    epoch_losses = []
    epoch_losses_pred_loss = []
    used_indices = list()
    unused_indices = torch.randperm(len(train_dataset.tensors))

    while len(unused_indices) > 2:
        if batch_num < 1 and epoch < 1:
        # if batch_num < 10000:
            batch_indices = unused_indices[:batch_size]
            unused_indices = unused_indices[batch_size:]
            used_indices.extend(batch_indices.tolist())
            # print('random')
        else:
            with torch.no_grad():
                # print('loss_pred')
                unused_samples = torch.utils.data.Subset(train_dataset.tensors, unused_indices)
                unused_loader = torch.utils.data.DataLoader(unused_samples, batch_size=len(unused_samples),
                                                            shuffle=False)
                unused_inputs = next(iter(unused_loader))

                _, features = model(unused_inputs)
                loss_scores = model_loss(features).squeeze()
                loss_scores_glob = loss_scores
                # Convert loss scores to probabilities
                loss_probs = torch.softmax(loss_scores, dim=0)

                # Sample indices based on loss probabilities
                if len(loss_probs) >= batch_size:
                    batch_indices = torch.multinomial(loss_probs, batch_size, replacement=False)
                else:
                    batch_indices = torch.arange(len(loss_probs))

                batch_indices = unused_indices[batch_indices]
                # Update used and unused indices
                used_indices.extend(batch_indices.tolist())
                unused_indices = np.array([i for i in unused_indices if i not in batch_indices])

        if len(batch_indices) == 0:
            break

        selected_dataset = torch.utils.data.Subset(train_dataset.tensors, batch_indices)
        selected_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, batch_inputs in enumerate(selected_loader):
            optimizer.zero_grad()
            optimizer_loss.zero_grad()
            outputs, features = model(batch_inputs)
            outputs_loss = model_loss(features)
            batch_outputs = compute_result(batch_inputs[:, 0], batch_inputs[:, 0])
            batch_outputs = batch_outputs.unsqueeze(1)
            loss = criterion(outputs, batch_outputs)
            loss_pred_loss = criterion(outputs_loss, loss)

            global_loss = loss.mean() + loss_pred_loss.mean()

            global_loss.backward()

            optimizer.step()
            optimizer_loss.step()

            epoch_losses.append(loss.mean().item())
            epoch_losses_pred_loss.append(loss_pred_loss.mean().item())

        # os.makedirs('models', exist_ok=True)
        # torch.save(model.state_dict(), f'models/model_batch{batch_num}_epoch{epoch + 1}.pth')
        # torch.save(model_loss.state_dict(), f'models/model_loss_batch{batch_num}_epoch{epoch + 1}.pth')

        batch_num += 1

    losses.append(np.mean(epoch_losses))
    losses_pred_loss.append(np.mean(epoch_losses_pred_loss))

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_losses):.4f}, Loss prediction Loss: {np.mean(epoch_losses_pred_loss):.4f}')

plt.clf()
plt.plot(losses, label='Training MSE Loss')
plt.plot(losses_pred_loss, label='Loss Prediction Loss')
# plt.plot(test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()