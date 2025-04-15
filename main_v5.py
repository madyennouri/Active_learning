import numpy as np
import torch.optim as optim
from code_execution import label_inputs

from models import *
from utils import *

# Load the configuration
config = load_config('config_v5.json')

# Access to parameters
num_inputs          = config['num_inputs']
resolution          = config['resolution']
input_max           = config['input_max']
input_min           = config['input_min']
batch_size          = config['batch_size']
epoch_size          = config['epoch_size']
max_iter            = config['max_iter']
max_labels          = config['max_labels']
num_epochs_per_iter = config['num_epochs_per_iter']

inputs = build_inputs(num_inputs, resolution, input_max, input_min)

input_scaler = MinMaxScaler()
inputs = input_scaler.fit_transform(inputs)
inputs = torch.tensor(inputs, dtype=torch.float32)
train_dataset = IndexedTensorDataset(inputs)

train_loader = torch.utils.data.DataLoader(train_dataset.tensors, batch_size=batch_size, shuffle=False)

model = SimpleNN()
model_loss = LossNet()

criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer_loss = optim.SGD(model_loss.parameters(), lr=0.01)


losses = []
losses_pred_loss = []
loss_scores_glob = [1]

losses_vector = torch.ones(len(train_dataset.tensors))
predicted_losses_vector = torch.ones(len(train_dataset.tensors))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Initialize lists to store training and test errors
test_errors = []

iter_num          = 0
labeled_indices   = list()
unlabeled_indices = torch.randperm(len(train_dataset.tensors))
labeled_inputs = torch.empty((0, num_inputs))
labels = torch.empty(0)

while len(labeled_indices) < max_labels:
    if iter_num < 3:
        batch_indices = unlabeled_indices[:batch_size]
        unlabeled_indices = unlabeled_indices[batch_size:]
        labeled_indices.extend(batch_indices.tolist())
    else:
        with torch.no_grad():
            unlabeled_samples = torch.utils.data.Subset(train_dataset.tensors, unlabeled_indices)
            unlabeled_loader = torch.utils.data.DataLoader(unlabeled_samples, batch_size=len(unlabeled_samples),
                                                        shuffle=False)
            unlabeled_inputs = next(iter(unlabeled_loader))

            _, features = model(unlabeled_inputs)
            loss_scores = model_loss(features).squeeze()
            loss_scores_glob = loss_scores
            # Convert loss scores to probabilities
            loss_probs = torch.softmax(loss_scores, dim=0)

            # Sample indices based on loss probabilities
            if len(loss_probs) >= batch_size:
                batch_indices = torch.multinomial(loss_probs, batch_size, replacement=False)
            else:
                batch_indices = torch.arange(len(loss_probs))

            batch_indices = unlabeled_indices[batch_indices]
            # Update labeled and unlabeled indices
            labeled_indices.extend(batch_indices.tolist())
            unlabeled_indices = np.array([i for i in unlabeled_indices if i not in batch_indices])

    if len(batch_indices) == 0:
        break

    new_inputs = inputs[batch_indices, :]
    new_label = label_inputs(inputs[batch_indices, 0], inputs[batch_indices, 1])

    labeled_inputs = torch.cat((labeled_inputs, new_inputs), dim=0)
    labels = labels.squeeze()
    labels = torch.cat((labels, new_label), dim=0)

    if len(labeled_indices) <= epoch_size:
        epoch_indices = np.arange(len(labeled_indices))
    else:
        optimizer.zero_grad()
        outputs, features = model(labeled_inputs)
        labels = labels.unsqueeze(1)
        loss_pick = criterion(outputs, labels)
        loss_probs_pick = torch.softmax(loss_pick, dim=0)
        loss_probs_pick = loss_probs_pick.squeeze()
        epoch_indices = torch.multinomial(loss_probs_pick, epoch_size, replacement=False)

    batch_num = 0
    model.train()
    model_loss.train()

    epoch_losses = []
    epoch_losses_pred_loss = []

    selected_dataset = torch.utils.data.Subset(labeled_inputs, epoch_indices)
    selected_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, batch_inputs in enumerate(selected_loader):
        optimizer.zero_grad()
        optimizer_loss.zero_grad()
        outputs, features = model(batch_inputs)
        outputs_loss = model_loss(features)
        batch_outputs = labels[batch_idx]
        loss = criterion(outputs, batch_outputs)
        loss_pred_loss = criterion(outputs_loss, loss)

        global_loss = loss.mean() + loss_pred_loss.mean()

        global_loss.backward()

        optimizer.step()
        optimizer_loss.step()

        epoch_losses.append(loss.mean().item())
        epoch_losses_pred_loss.append(loss_pred_loss.mean().item())

        batch_num += 1

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/model_batch{iter_num}.pth')
    torch.save(model_loss.state_dict(), f'models/model_loss_batch{iter_num}.pth')


    losses.append(np.mean(epoch_losses))
    losses_pred_loss.append(np.mean(epoch_losses_pred_loss))

    iter_num += 1
    print(
        f'Iteration [{iter_num}], Loss: {np.mean(epoch_losses):.4f}, Loss prediction Loss: {np.mean(epoch_losses_pred_loss):.4f}')

os.makedirs('loss', exist_ok=True)
np.savetxt('losses.txt', losses, fmt='%.6f')
np.savetxt('losses_pred_loss.txt', losses_pred_loss, fmt='%.6f')

print(len(labeled_indices))
plt.clf()
plt.plot(losses, label='Training MSE Loss')
plt.plot(losses_pred_loss, label='Loss Prediction Loss')
# plt.plot(test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()