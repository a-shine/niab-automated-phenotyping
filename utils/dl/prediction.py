import torch

def predict_with_uncertainty(model, image, device, n_times=200):
    # Set model to training mode to enable dropout at test time
    model.train()

    # List to store predictions
    predictions = []

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
