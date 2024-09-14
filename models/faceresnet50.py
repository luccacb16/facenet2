import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50
import warnings

class FaceResNet50(nn.Module):
    def __init__(self, n_classes=0, emb_size=256):
        super(FaceResNet50, self).__init__()
        resnet = resnet50()

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Linear(2048, emb_size)

        #self._initialize_weights()
        
        self.emb_size = emb_size
        self.n_classes = n_classes
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        #x = F.normalize(x, p=2, dim=1) # L2
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def save_checkpoint(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
            
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in self.state_dict().items()}
        
        checkpoint = {
            'state_dict': model_state_dict,
            'n_classes': self.n_classes,
            'emb_size': self.emb_size
        }
        torch.save(checkpoint, os.path.join(path, filename))
        
    def load_checkpoint(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(path)

        model = FaceResNet50(n_classes=0, emb_size=checkpoint['emb_size'])

        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'fc2' not in k}
        model.load_state_dict(state_dict, strict=False)

        return model

    def freeze(self):
        # Congela todos os parâmetros das camadas, exceto 'fc1'
        for name, param in self.named_parameters():
            if 'fc1' not in name:
                param.requires_grad = False