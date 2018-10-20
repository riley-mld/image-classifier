import numpy as np

import argparse
from time import time
import json

import torch
from torchvision import transforms, datasets, models
from torch import nn
from torch import optim
import torch.nn.functional as F

import functions as func


def main():
    start_time = time()
    
    # Get comand line arguments.
    in_arg = get_input_args()
    
    # Create datasets and dataloaders from the image dataset path.
    trainloader, validationloader, testloader, train_dataset, validation_dataset, test_dataset = prepare_datasets(in_arg.data_dir)
    
    # Get category label names from the file.
    cat_to_name = get_labels(in_arg.cat_file)
    
    # Cretes the model based on comand line arguments and dataset.
    model, criterion, optimizer, input_units, output_units, hidden_units = make_model(in_arg, cat_to_name)
    # Set device.
    device = set_device(in_arg.gpu)
    
    # Print command line arguments.
    print_command_line_argument(in_arg, device)

    epochs = in_arg.epochs
    print_every = 40
    
    # Train the model.

    model = func.deep_learning(model, trainloader, validationloader, criterion, optimizer, epochs, print_every, device)
    
    # Check accuracy on test dataset.
    func.check_accuracy_on_test(model, testloader, optimizer, criterion, device)
    
    # Save the model.
    save_model(in_arg, input_units, output_units, hidden_units, epochs, print_every, model, optimizer, test_dataset)

    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

def get_input_args():
    """ Get the command line arguments. """
    # Creates parse 
    parser = argparse.ArgumentParser()


    parser.add_argument('data_dir', type=str, 
                        help='path to folder of datasets')
    parser.add_argument('cat_file', type=str, default='cat_to_name.json',
                        help='path to category file')
    parser.add_argument('--arch', type=str, default='resnet152', 
                        help='chosen model')
    parser.add_argument('--save_dir', type=str, default='test1.pth',
                        help='set directory for save file')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--hidden_units','--list', nargs='+', default=['1000'],
                        help='Setting number of hidden layers and their units in the architecture')
    parser.add_argument('--gpu', type=str, default='cpu', 
                        help='set device for training.')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='set number of epochs.')
    
    # returns parsed argument collection
    return parser.parse_args()


def print_command_line_argument(in_arg, device):
    """Print Command Line Arguments"""
    print("Command Line Arguments:\n     data_dir =", in_arg.data_dir, 
          "\n     arch =", in_arg.arch, "\n     save_dir =", in_arg.save_dir, "\n     learning_rate =", in_arg.learning_rate,
          "\n     hidden_units =", in_arg.hidden_units, "\n     gpu = ", device,
          "\n     epochs =", in_arg.epochs, "\n     cat_to_name =", in_arg.cat_file)
    
    
def prepare_datasets(data_directory):
    """Prepare datasets and train loader."""
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define Transformations
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return trainloader, validationloader, testloader, train_dataset, validation_dataset, test_dataset


def get_labels(file):
    """Get Label files."""
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name


def set_device(gpu):
    """Set device."""
    if gpu.lower() == 'cpu':
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return device
    
    
def save_model(in_arg, input_units, output_units, hidden_units, epochs, print_every, model, optimizer, test_dataset):
    """Save the model."""
    if in_arg.arch.lower() == 'resnet' or in_arg.arch.lower() == 'resnet152':        
        checkpoint = {'input_size': input_units,
                      'output_size': output_units,
                      'hidden_size': hidden_units,
                      'model': 'models.resnet152(pretrained=True)',
                      'drop_p': 0.5,
                      'criterion': 'nn.NLLLoss()',
                      'optimizer': 'optim.Adam(model.fc.parameters(), lr=0.001)',
                      'epochs': epochs,
                      'print_every': print_every,
                      'class_to_idx': test_dataset.class_to_idx,
                      'device': 'torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
                      'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
    
    elif in_arg.arch.lower() == 'vgg' or in_arg.arch.lower() == 'vgg19':
         checkpoint = {'input_size': input_units,
                      'output_size': output_units,
                      'hidden_size': hidden_units,
                      'model': 'models.vgg19(pretrained=True)',
                      'drop_p': 0.5,
                      'criterion': 'nn.NLLLoss()',
                      'optimizer': 'optim.Adam(model.classifier.parameters(), lr=0.001)',
                      'epochs': epochs,
                      'print_every': print_every,
                      'class_to_idx': test_dataset.class_to_idx,
                      'device': 'torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
                      'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
            
    torch.save(checkpoint, in_arg.save_dir)

 
def make_model(in_arg, cat_to_name):
    """Create Model"""
    if in_arg.arch.lower() == 'resnet' or in_arg.arch.lower() == 'resnet152':
        model = models.resnet152(pretrained=True)
        
        input_units = model.fc.in_features
        output_units = len(cat_to_name)
        hidden_units = [int(i) for i in in_arg.hidden_units]
        
        classifier = func.Network(input_units, output_units, hidden_units, drop_p=0.5)
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.fc = classifier
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=in_arg.learning_rate)
        
    elif in_arg.arch.lower() == 'vgg' or in_arg.arch.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
        
        input_units = model.classifier.in_features
        output_units = len(cat_to_name)
        hidden_units = [int(i) for i in in_arg.hidden_units]
        
        classifier = func.Network(input_units, output_units, hidden_units, drop_p=0.5)
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = classifier
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
        
    return model, criterion, optimizer, input_units, output_units, hidden_units
    
main()