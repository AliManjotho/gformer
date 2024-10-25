import torch


def calculate_gram_matrix(layer_output):
    """
    Calculate the Gram matrix for a given layer output.
    
    Args:
        layer_output (torch.Tensor): The output of a layer with shape (N, C, H, W)
                                     where N is the batch size, C is the number of channels,
                                     H is the height, and W is the width.
    
    Returns:
        torch.Tensor: The Gram matrix of shape (C, C).
    """
    # Flatten the layer output to shape (N, C, H*W)
    N, C, H, W = layer_output.size()
    features = layer_output.view(N, C, H * W)
    
    # Normalize the features
    features = features / (C * H * W)
    
    # Calculate the Gram matrix
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize the Gram matrix
    gram_matrix = gram_matrix / (C * H * W)
    
    return gram_matrix
