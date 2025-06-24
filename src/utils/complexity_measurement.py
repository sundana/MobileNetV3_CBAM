import torch.nn as nn
from ptflops import get_model_complexity_info
import pandas as pd

def calculate_flops(models: list[nn.Module]) -> pd.DataFrame:
    """Calculate model FLOPs

    Args:
        models (list(nn.Module)): PyTorch model based on nn.Module

    Returns:
        pd.DataFrame: table of model complexity
    """
    model_complexity = {
        "models": [],
        "flops": [],
        "params": []
    }
    for model in models:
        model = model(num_classes=29)
        flop, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        model_complexity["models"].append(model.__class__.__name__)
        model_complexity["flops"].append(flop)
        model_complexity["params"].append(params)

    df = pd.DataFrame(model_complexity)
    return df


if __name__ == "__main__":
    from models import mobilenetv3, proposed_model
    
    # Register model
    models = [ 
        mobilenetv3.MobileNetV3_Large, 
        proposed_model.MobileNetV3_Large_CBAM_16,
        proposed_model.MobileNetV3_Large_CBAM_32,
        mobilenetv3.MobileNetV3_Small, 
        proposed_model.MobileNetV3_Small_CBAM_16,
        proposed_model.MobileNetV3_Small_CBAM_32,
    ]

    df = calculate_flops(models)
    df.to_csv('./results/complexity/model_complexity.csv', index=False) # Save result csv
    print("Saving model complexity completed to ./results/complexity")
    print(df)