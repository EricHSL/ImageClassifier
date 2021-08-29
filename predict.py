import argparse
import numpy as np
import torch
import json
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn, optim
from PIL import Image

#flowers/test/1/image_06743.jpg checkpoint.pth
structures = {"vgg16" : 25088,
             "densenet121" : 1024}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='flowers/test/1/image_06752.jpg', nargs = '?', action="store", type=str)
    parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
    parser.add_argument('--top_k', dest="top_k", default=5, action="store", type=int)
    parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
    parser.add_argument('--gpu', action="store", dest="gpu", default='gpu')
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action='store', help='checkpoint file')

    return parser.parse_args()

def setup_predict_network(arch="vgg16", dropout=0.1, hidden_units=4096, lr=0.001, device='gpu'):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    #switch = {'densenet121' : models.densenet121(pretrained=True),
    #          'vgg16':models.vgg16(pretrained=True)}
    #model = switch[arch]
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.classifier = nn.Sequential(nn.Linear(structures[arch], hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1)
                                    )
    model = model.to('cuda')
    #criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    
    return model

def load_checkpoint(filepath = 'checkpoint.pth'):
    checkpoint = torch.load(filepath)
    lr = checkpoint['learning_rate']
    hiddenUnits = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['epochs']
    arch = checkpoint['arch']
    
    model = setup_predict_network(arch, dropout, hiddenUnits, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    perform_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    #img = Image.open(image)
    #image = perform_transforms(img)
    
    image = perform_transforms(Image.open(image))
    return image

def predict(imagePath, model, topk=5, device='gpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = model.eval()
    
    #img = process_image(image_path)
    #img = img.numpy()
    #img = torch.from_numpy(np.array([img])).float()
    img = torch.from_numpy(np.array([process_image(imagePath).numpy()])).float()
        
    with torch.no_grad():
        output = model.forward(img.cuda())
    
    probability = torch.exp(output).data
    
    return probability.topk(topk)

def main():
    args = get_args()
    path = args.checkpoint
    model = load_checkpoint(path)
    #class_to_idx = model.class_to_idx.item()
    
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    
    probabilities = predict(args.input, model, args.top_k, args.gpu)
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping [item] for item in probabilities[1][0].cpu().numpy()]
    
    labels = [cat_to_name[str(index)] for index in classes]
    
    probability = np.array(probabilities[0][0])
    
    i = 0
    while i < args.top_k:
        print("{}. {} / Probability: {:.2f}%".format((i+1), labels[i].capitalize(), 100*probability[i]))
        i += 1
    print("--Prediction Fin.!!")
    

if __name__ == "__main__":
    main()
