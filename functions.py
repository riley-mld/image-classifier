import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import torch
from torchvision import transforms, datasets, models
from torch import nn
from torch import optim
import torch.nn.functional as F


class Network(nn.Module):
    """Neural Network"""
    def __init__(self, input_units, output_units, hidden_units, drop_p=0.5):
        """Initialize Network."""
        
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_units, hidden_units[0])])
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_units[-1], output_units)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        """forward pass"""
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    

def validation(model, validationloader, optimizer, criterion, device):
    """Validation pass."""
    test_loss = 0
    accuracy = 0
    correct = 0
    total = 0
    
    for inputs, labels in validationloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        ps = torch.exp(outputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = 100 * correct / total
    
    return test_loss, accuracy


def deep_learning(model, trainloader, validationloader,criterion, optimizer, epochs, print_every, device):
    """Trains the neural network."""
    epochs = epochs
    print_every = print_every
    steps = 0
    
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            # Transfering tensors to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validationloader, optimizer, criterion, device)
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Training Loss: {:.3f}...".format(running_loss/print_every),
                      "Test Loss: {:.3f}...".format(test_loss/len(validationloader)),
                      "Test Accuracy: {:.3f}...".format(accuracy))
                
                running_loss = 0
    
    return model


def check_accuracy_on_test(model, testloader, optimizer, criterion, device):
    """Check accuracy on the test data."""
    correct = 0
    total = 0
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.forward(inputs)
            ps = torch.exp(outputs)
            _, predicted = torch.max(ps.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum().item()
            
        print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
        print(correct)
        print(total)

    
def load_checkpoint(filepath):
    """Load and return a model from a checkpoint"""
    checkpoint = torch.load(filepath)
    
    model = eval(checkpoint['model'])
    classifier = Network(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_size'], drop_p=checkpoint['drop_p'])
    
    model.fc = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    device = eval(checkpoint['device'])
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    print_every = checkpoint['print_every']
    optimizer = eval(checkpoint['optimizer'])
    criterion = eval(checkpoint['criterion'])
    optimizer.state_dict = checkpoint['optimizer_state_dict']
    
    return model, device, epochs, print_every, optimizer, criterion


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model."""
    
    pil_image = Image.open(image)
    pil_image = pil_image.resize((256, 256))
    pil_image = pil_image.crop((16, 16, 240, 240))
    
    np_image = np.array(pil_image)
    np_image = np_image.astype(np.float64)
    np_image /= 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((1, 0, 2))
    np_image = np_image.transpose()
    
    return np_image


def predict(image_path, model, device, topk=5):
    """Predict the class of an image with trained model."""
    
    model_load, device, epochs, print_every, optimizer, criterion = load_checkpoint(model)
    
    image = process_image(image_path)
    inputs = torch.from_numpy(image)
    inputs.unsqueeze_(0)
    inputs = inputs.type(torch.FloatTensor)
    inputs = inputs.to(device)

    
    model_load.to(device)
    model_load.eval()
    with torch.no_grad():
        outputs = model_load.forward(inputs)
        outputs = torch.exp(outputs)
        _, predicted = torch.max(outputs.data, 1)
        ps = outputs.topk(topk)
        classes = []
        for idx in ps[1].data.cpu().numpy()[0]:
            classes.append([c for c, i in model_load.class_to_idx.items() if i == idx][0])
            
    return  ps[0].data.cpu().numpy()[0], classes