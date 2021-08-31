#This file checks if checkpoint.pth works...

import torch
import torchvision.models as models

def load_checkpoint(filepath = 'checkpoint.pth'):
    checkpoint = torch.load(filepath)
    #arch = checkpoint['arch']
    #if arch == 'vgg16':
    #    model = models.vgg16(pretrained=True)
    #elif arch =='densenet121':
    #    model = models.densenet121(pretrained=True)
        
    #for param in model.parameters():
    #    param.requires_grad = False
    
    #model.classifier = checkpoint['classifier']
    #model.load_state_dict(checkpoint['state_dict'])
    
    return checkpoint

def main():
    
    print(load_checkpoint())

if __name__ == "__main__":
    main()
