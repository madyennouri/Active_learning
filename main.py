import torch.optim as optim
import os

from models import *
from utils import *

filepath = 'C:/Users/madyen/OneDrive - University of Arizona/Desktop/Storage_UA/1-Work/IV- Plate impact/automated_impact/results/strain_rate_max_data.txt'
inputs, outputs = data_loader(filepath)

x_train, y_train, x_test, y_test, input_scaler, output_scaler = slip_and_scale(inputs, outputs)

train_dataset = IndexedTensorDataset((x_train, y_train))
test_dataset = IndexedTensorDataset((x_test, y_test))

batch_size = 3
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

plot_3D_scatter(x_train[:, 0], x_train[:, 1], y_train[:])

model = SimpleNN()
model_loss = LossNet()

criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer_loss = optim.SGD(model_loss.parameters(), lr=0.01)

num_epochs = 10
losses = []
losses_pred_loss = []

losses_vector = torch.ones(len(train_dataset))
predicted_losses_vector = torch.ones(len(train_dataset))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))


# Define the grid for the heatmap
grid_size = 10
xi = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), grid_size)
yi = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), grid_size)
xi, yi = np.meshgrid(xi, yi)

# Initialize lists to store training and test errors
test_errors = []

for epoch in range(num_epochs):
    batch_num = 0
    model.train()
    model_loss.train()

    epoch_losses = []
    epoch_losses_pred_loss = []
    used_indices = []
    unused_indices = list(range(len(train_dataset)))

    if epoch == 0:
        # First epoch: Randomly select batches
        indices = torch.randperm(len(train_dataset))
    else:
        # Subsequent epochs: Initialize the used_indices
        used_indices = []

    while len(unused_indices) > 0:
        if epoch == 0:
            batch_indices = indices[:batch_size]
            indices = indices[batch_size:]
        else:
            # Apply model_loss to select top samples from unused_indices
            with torch.no_grad():
                unused_samples = torch.utils.data.Subset(train_dataset, unused_indices)
                unused_loader = torch.utils.data.DataLoader(unused_samples, batch_size=len(unused_samples),
                                                            shuffle=False)
                batch_inputs, _, batch_indices = next(iter(unused_loader))
                _, features = model(batch_inputs)
                loss_scores = model_loss(features).squeeze()

                # Convert loss scores to probabilities
                loss_probs = torch.softmax(loss_scores, dim=0)

                # Sample indices based on loss probabilities
                if len(loss_probs) >= batch_size:
                    sampled_indices = torch.multinomial(loss_probs, batch_size, replacement=False)
                else:
                    sampled_indices = torch.arange(len(loss_probs))

                # Retrieve the sampled batch indices
                batch_indices = batch_indices[sampled_indices]

                # Update used and unused indices
                used_indices.extend(batch_indices.tolist())
                unused_indices = [i for i in unused_indices if i not in batch_indices.tolist()]

        if len(batch_indices) == 0:
            break

        # Create DataLoader for selected indices
        selected_dataset = torch.utils.data.Subset(train_dataset, batch_indices)
        selected_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, (batch_inputs, batch_outputs, batch_indices) in enumerate(selected_loader):
            optimizer.zero_grad()
            optimizer_loss.zero_grad()
            batch_num += 1
            outputs, features = model(batch_inputs)
            outputs_loss = model_loss(features)

            loss = criterion(outputs, batch_outputs)
            loss_pred_loss = criterion(outputs_loss, loss)

            global_loss = loss.mean() + loss_pred_loss.mean()

            global_loss.backward()

            optimizer.step()
            optimizer_loss.step()

            epoch_losses.append(loss.mean().item())
            epoch_losses_pred_loss.append(loss_pred_loss.mean().item())

            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/model_batch{batch_num}_epoch{epoch + 1}.pth')
            torch.save(model_loss.state_dict(), f'models/model_loss_batch{batch_num}_epoch{epoch + 1}.pth')

    losses.append(np.mean(epoch_losses))
    losses_pred_loss.append(np.mean(epoch_losses_pred_loss))



    # Calculate and store the test error for the current epoch
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(x_test)
        test_loss = criterion(test_outputs, y_test)
        test_errors.append(test_loss.mean().item())

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(epoch_losses):.4f}, Loss prediction Loss: {np.mean(epoch_losses_pred_loss):.4f}, Test Error: {test_errors[-1]:.4f}')
np.savetxt("./results/losses_ours.txt", losses, fmt="%.6f")

plt.clf()
plt.plot(losses, label='Training MSE Loss')
plt.plot(losses_pred_loss, label='Loss Prediction Loss')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
