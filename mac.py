import torch
import torch.nn as nn

def calculate_layer_mac(layer, input_shape):
    """
    Calculate the Memory Access Cost (MAC) of a single convolutional layer.

    Args:
        layer (nn.Module): The convolutional layer.
        input_shape (tuple): Shape of the input to the layer (C, H, W).

    Returns:
        float: The MAC value for the given layer.
    """
    if not isinstance(layer, nn.Conv2d):
        return 0  

    Cin, H, W = input_shape
    Cout = layer.out_channels
    kernel_size = layer.kernel_size[0]  

    stride = layer.stride[0]
    padding = layer.padding[0]
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    feature_maps_memory = H_out * W_out * (Cin + Cout)
    weights_memory = kernel_size * kernel_size * (Cin * Cout)
    mac = feature_maps_memory + weights_memory

    return mac, (Cout, H_out, W_out)

def calculate_total_mac(model, input_shape):
    """
    Calculate the total Memory Access Cost (MAC) for all layers in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        input_shape (tuple): Shape of the input tensor (C, H, W).

    Returns:
        float: The total MAC value for the model.
    """
    total_mac = 0
    current_shape = input_shape

    for layer in model.children():
        if isinstance(layer, nn.Sequential):  
            total_mac += calculate_total_mac(layer, current_shape)
        else:
            mac, output_shape = calculate_layer_mac(layer, current_shape)
            total_mac += mac
            current_shape = output_shape  

    return total_mac
