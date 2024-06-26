import torch


def predict_with_uncertainty(model, image, device, n_times=500):
    """
    Predict with uncertainty using Monte Carlo dropout method.

    Args:
        model: PyTorch model
        image: PyTorch tensor representing the image
        n_times: Number of times to run the prediction with dropout enabled

    Returns:
        mean: Mean of the predictions
        variance: Variance of the predictions
    """
    model.train()  # Set model to training mode to enable dropout at test time

    predictions = []  # Store predictions

    # Run prediction n_times with dropout enabled to variability in the
    # predictions
    with torch.no_grad():
        for _ in range(n_times):
            output = model(image.unsqueeze(0).to(device))
            predictions.append(output)

    # Convert predictions list to tensor
    predictions = torch.stack(predictions)

    # Calculate mean and variance
    mean = torch.mean(predictions, dim=0)
    variance = torch.var(predictions, dim=0)

    return mean, variance
