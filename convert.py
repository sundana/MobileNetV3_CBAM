import torch
import torchvision.models as models

model = models.mobilenet_v3_small(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "./converted_models/mobilenet_v3_small.onnx")

