import argparse
import torch
from torch import nn, optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import json


structures = {"vgg16" : 25088,
             "densenet121" : 1024}

def get_args():
    parser = argparse.ArgumentParser(description='Parser for train.py')
    parser.add_argument('data_dir', action="store", default="./flowers/")
    parser.add_argument('--save_dir', action="store", default='./checkpoint.pth')
    parser.add_argument('--arch', action = 'store', default='vgg16')
    parser.add_argument('--learning_rate', action='store', default=0.001, type=float)
    parser.add_argument('--hidden_units', action='store', dest="hidden_units", default=512, type=int)
    parser.add_argument('--epochs', action='store', default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.5)
    parser.add_argument('--gpu', default="gpu", action='store')
    return parser.parse_args()

def setup_network(arch="vgg16", dropout=0.1, hidden_units=4096, lr=0.001, device='gpu'):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.classifier = nn.Sequential(nn.Linear(structures[arch], hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1)
                                    )
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    
    return model, criterion

def main():
    args = get_args()
    
    if args.gpu == 'gpu':
        device='cuda'
    else:
        device='cpu'
        
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    dataDir = args.data_dir
    trainDir = dataDir + '/train'
    testDir = dataDir + '/test'
    validDir = dataDir + '/valid'
    
    #Define transforms
    mean = [0.485, 0.456, 0.406]
    stdev = [0.229, 0.224, 0.225]
    
    trainTransforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, stdev)])
    
    #valid_transforms and test_transforms have the same format
    testTransforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, stdev)])
    validTransforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, stdev)])
    
    #Load datasets with ImageFolder
    dataset = {'train' : datasets.ImageFolder(trainDir, transform=trainTransforms),
               'test' : datasets.ImageFolder(testDir, transform=testTransforms),
               'valid' : datasets.ImageFolder(validDir, transform=validTransforms)}
    
    
    #Define dataloaders
    dataloaders = {'train' : DataLoader(dataset['train'], batch_size=32, shuffle=True),
                   'test' : DataLoader(dataset['test'], batch_size=32),
                   'valid' : DataLoader(dataset['valid'], batch_size=32)}
    
    model, criterion = setup_network(args.arch, args.dropout, args.hidden_units, args.learning_rate, args.gpu)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    #train model
    steps = 0
    runningLoss = 0
    printEvery = 15
    start = time.time()
    
    print("\n--Training starting:")
    for epoch in range(args.epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()
    
            if (steps % printEvery) == 0:
                validLoss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        
                        logps = model.forward(inputs)
                        batchLoss = criterion(logps, labels)
                        validLoss += batchLoss.item()
                        
                        ps = torch.exp(logps)
                        top_p, topClass = ps.topk(1, dim=1)
                        equals = topClass == labels.view(*topClass.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Training Loss: {runningLoss/len(dataloaders['test']):.3f}.. "
                      f"Validation Loss: {validLoss/len(dataloaders['valid']):.3f}.. "
                      f"Validation Accuracy: {100 * accuracy/len(dataloaders['valid']):.3f}%")
                runningLoss = 0
                model.train()  
                
    timeEnd = time.time() - start
    print("--Training Completed!!")
    print("--Time Elapsed: {}m {}s".format(int(timeEnd // 60), int(timeEnd % 60)))
    
    model.to(device)
    model.class_to_idx = dataset['train'].class_to_idx
    torch.save({'arch' : args.arch,
                'hidden_units' : args.hidden_units,
                'dropout' : args.dropout,
                'learning_rate' : args.learning_rate,
                'epochs' : args.epochs,
                'state_dict' : model.state_dict(),
                'class_to_idx' : model.class_to_idx}, args.save_dir)
    print("--Checkpoint saved")

    

if __name__ == "__main__":
    main()
   
          
