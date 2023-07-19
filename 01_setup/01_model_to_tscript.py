import torch
import torchvision

# An instance of your model with pretrained ImageNet weights.
model = torchvision.models.resnet34(pretrained=True)
model.fc = torch.nn.Linear(512, 6)

# Save the model with pretrained weights
torch.save(model.state_dict(), "model_with_pretrained_weights.bin")

# Load the saved model
model.load_state_dict(torch.load("model_with_pretrained_weights.bin", map_location=torch.device('cpu')))

# Switch the model to eval mode
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("traced_model.pt")
