import torch
from my_models.proposedmodel import MobileNetV3_Large

model = MobileNetV3_Large(29)
model.load_state_dict(torch.load(
    './model_checkpoints/proposed_model_large_16.pth', map_location='cpu'))
model.eval()


scripted_model = torch.jit.script(model)
scripted_model.save('./converted_models/proposedmodel_large_16.pt')
