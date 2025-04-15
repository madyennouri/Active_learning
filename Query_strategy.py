import torch
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from torch.nn.functional import softmax, cross_entropy, mse_loss


def uncertainty_sampling(model, x, batch_size, strategy="least_confidence", device="cpu"):
    """
    Uncertainty sampling strategies for PyTorch models.
    Arguments:
        model: A PyTorch model.
        x: Input data, can be a NumPy array or PyTorch tensor.
        batch_size: Number of samples to query.
        strategy: "least_confidence", "margin", or "entropy".
        device: Device for computations ("cpu" or "cuda").
    Returns:
        Indices of the selected batch.
    """
    model.eval()

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    elif not isinstance(x, torch.Tensor):
        raise TypeError("Input 'x' must be a NumPy array or PyTorch tensor.")

    x = x.to(device)

    with torch.no_grad():
        outputs, _ = model(x)
        probabilities = softmax(outputs).cpu().numpy()

    if strategy == "least_confidence":
        scores = 1 - np.max(probabilities, axis=1)
    elif strategy == "margin":
        sorted_probs = np.sort(probabilities, axis=1)
        scores = sorted_probs[:, -1] - sorted_probs[:, -2]
    elif strategy == "entropy":
        scores = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    else:
        raise ValueError("Invalid strategy. Choose from 'least_confidence', 'margin', or 'entropy'.")
    samples_idx = np.argsort(scores)[-batch_size:][::-1]
    return samples_idx.tolist()


def random_sampling(x, batch_size):
    """
    Random sampling strategy.
    Arguments:
        x: Input data, can be a NumPy array or PyTorch tensor.
        batch_size: Number of samples to query.
    Returns:
        Indices of the selected batch.
    """
    if isinstance(x, np.ndarray):
        num_samples = x.shape[0]
    elif isinstance(x, torch.Tensor):
        num_samples = x.size(0)
    else:
        raise TypeError("Input 'x' must be a NumPy array or PyTorch tensor.")

    samples_idx = np.random.choice(num_samples, size=batch_size)
    return samples_idx.tolist()


def density_weighted_sampling(model, x, batch_size, device="cpu"):
    """
    Density-weighted sampling strategy.
    Arguments:
        model: A PyTorch model.
        x: Input data, can be a NumPy array or PyTorch tensor.
        batch_size: Number of samples to query.
        device: Device for computations ("cpu" or "cuda").
    Returns:
        Indices of the selected batch.
    """
    model.eval()

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    elif not isinstance(x, torch.Tensor):
        raise TypeError("Input 'x' must be a NumPy array or PyTorch tensor.")

    x = x.to(device)

    with torch.no_grad():
        embeddings, _ = model(x)

    embeddings = embeddings.cpu().numpy()

    similarity_matrix = cosine_similarity(embeddings)

    density_scores = np.sum(similarity_matrix, axis=1)

    samples_idx = np.argsort(density_scores)[-batch_size:][::-1]
    return samples_idx.tolist()


def build_committee(x_labeled, y_labeled):
    """
    Builds a committee of models using Scikit-learn and the main PyTorch model.
    Arguments:
        x_labeled: Labeled input data (NumPy array or tensor).
        y_labeled: Corresponding labels (NumPy array).
    Returns:
        List of Scikit-learn models trained on labeled data.
    """
    if isinstance(x_labeled, torch.Tensor):
        x_labeled = x_labeled.numpy()
    if isinstance(y_labeled, torch.Tensor):
        y_labeled = y_labeled.numpy()

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    svm_model = SVR(kernel='linear')
    lr_model = Ridge(alpha=1.0, random_state=42)

    rf_model.fit(x_labeled, y_labeled)
    svm_model.fit(x_labeled, y_labeled)
    lr_model.fit(x_labeled, y_labeled)

    return [rf_model, svm_model, lr_model]

import numpy as np
from scipy.special import softmax

def query_by_committee_with_sklearn(model, committee_models, x, batch_size, device="cpu"):
    """
    Query-by-committee strategy using a PyTorch model and Scikit-learn models.
    Arguments:
        model: PyTorch model (main neural network).
        committee_models: List of Scikit-learn models (regression or classification).
        x: Input data (NumPy array or PyTorch tensor).
        batch_size: Number of samples to query.
        device: Device for computations ("cpu" or "cuda").
    Returns:
        Indices of the selected batch.
    """
    model.eval()

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    elif not isinstance(x, torch.Tensor):
        raise TypeError("Input 'x' must be a NumPy array or PyTorch tensor.")
    x = x.to(device)

    with torch.no_grad():
        outputs, _ = model(x)
        probabilities_nn = softmax(outputs.cpu().numpy(), axis=1)

    x_numpy = x.cpu().numpy()

    probabilities_sklearn = []
    for skl_model in committee_models:
        predictions = skl_model.predict(x_numpy)
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        normalized_predictions = softmax(predictions, axis=1)
        probabilities_sklearn.append(normalized_predictions)

    all_predictions = [probabilities_nn] + probabilities_sklearn
    all_predictions = np.array(all_predictions)  # Ensure it's a numerical array

    num_classes = probabilities_nn.shape[1]
    aligned_predictions = []
    for preds in all_predictions:
        if preds.shape[1] != num_classes:
            expanded_preds = np.zeros((preds.shape[0], num_classes))
            expanded_preds[:, 0] = preds[:, 0]
            aligned_predictions.append(expanded_preds)
        else:
            aligned_predictions.append(preds)
    aligned_predictions = np.stack(aligned_predictions, axis=0)

    avg_probabilities = np.mean(aligned_predictions, axis=0)
    vote_entropy = -np.sum(avg_probabilities * np.log(avg_probabilities + 1e-12), axis=1)

    samples_idx = np.argsort(vote_entropy)[-batch_size:][::-1]
    return samples_idx.tolist()

def estimated_error_reduction(model, x_unlabeled, x_labeled, y_labeled, batch_size, loss_type="classification", device="cpu"):
    """
    Estimated Error Reduction (EER) query strategy.
    Arguments:
        model: PyTorch model.
        x_unlabeled: Unlabeled data (NumPy array or PyTorch tensor).
        x_labeled: Labeled data (NumPy array or PyTorch tensor).
        y_labeled: Labels for labeled data (NumPy array or PyTorch tensor).
        batch_size: Number of samples to query.
        loss_type: "classification" or "regression".
        device: Device for computations ("cpu" or "cuda").
    Returns:
        Indices of the selected batch from the unlabeled set.
    """
    model.eval()

    if isinstance(x_labeled, np.ndarray):
        x_labeled = torch.tensor(x_labeled, dtype=torch.float32)
    if isinstance(y_labeled, np.ndarray):
        y_labeled = torch.tensor(y_labeled, dtype=torch.float32)

    if isinstance(x_unlabeled, np.ndarray):
        x_unlabeled = torch.tensor(x_unlabeled, dtype=torch.float32)

    x_labeled = x_labeled.to(device)
    y_labeled = y_labeled.to(device)
    x_unlabeled = x_unlabeled.to(device)

    with torch.no_grad():
        outputs = model(x_labeled)[0]
        if loss_type == "classification":
            baseline_loss = cross_entropy(outputs, y_labeled.long()).item()
        elif loss_type == "regression":
            baseline_loss = mse_loss(outputs, y_labeled).item()
        else:
            raise ValueError("Invalid loss_type. Choose 'classification' or 'regression'.")

    error_reductions = []
    for i in range(x_unlabeled.shape[0]):
        augmented_x = torch.cat([x_labeled, x_unlabeled[i].unsqueeze(0)])

        if loss_type == "classification":
            with torch.no_grad():
                pseudo_label = torch.argmax(model(x_unlabeled[i].unsqueeze(0))[0], dim=1)
            augmented_y = torch.cat([y_labeled, pseudo_label])
        elif loss_type == "regression":
            with torch.no_grad():
                pseudo_label = model(x_unlabeled[i].unsqueeze(0))[0]
            augmented_y = torch.cat([y_labeled, pseudo_label])

        with torch.no_grad():
            augmented_outputs = model(augmented_x)[0]
            if loss_type == "classification":
                augmented_loss = cross_entropy(augmented_outputs, augmented_y.long()).item()
            elif loss_type == "regression":
                augmented_loss = mse_loss(augmented_outputs, augmented_y).item()

        error_reduction = baseline_loss - augmented_loss
        error_reductions.append(error_reduction)

    error_reductions = np.array(error_reductions)
    top_indices = np.argsort(error_reductions)[-batch_size:][::-1]

    return top_indices.tolist()