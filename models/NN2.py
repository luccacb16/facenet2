import torch
import torch.nn as nn
import torch.nn.functional as F

'''
type             output_size depth #1x1 #3x3reduce #3x3  #5x5reduce #5x5  pool_proj_(p)
---------------------------------------------------------------------------------------
conv1 (7x7x3, 2) 112x112x64  1                                            
max pool + norm  56x56x64    0                                            m 3x3, 2
inception (2)    56x56x192   2          64         192                    
norm + max pool  28x28x192   0                                            m 3x3, 2
inception (3a)   28x28x256   2     64   96         128   16         32    m, 32p
inception (3b)   28x28x320   2     64   96         128   32         64    L2, 64p
inception (3c)   14x14x640   2     0    128        256,2 32         64,2  m 3x3,2
inception (4a)   14x14x640   2     256  96         192   32         64    L2, 128p
inception (4b)   14x14x640   2     224  112        224   32         64    L2, 128p
inception (4c)   14x14x640   2     192  128        256   32         64    L2, 128p
inception (4d)   14x14x640   2     160  144        288   32         64    L2, 128p
inception (4e)   7x7x1024    2     0    160        256,2 64         128,2 m 3x3,2
inception (5a)   7x7x1024    2     384  192        384   48         128   L2, 128p
inception (5b)   7x7x1024    2     384  192        384   48         128   m, 128p
avg pool         1x1x1024    0
fully conn       1x1x128     1
L2 normalization 1x1x128     0

Table 2. NN2. Details of the NN2 Inception incarnation. This model is almost identical to the one described in [16]. The two major
differences are the use of L2 pooling instead of max pooling (m), where specified. I.e. instead of taking the spatial max the L2 norm
is computed. The pooling is always 3x3 (aside from the final average pooling) and in parallel to the convolutional modules inside each
Inception module. If there is a dimensionality reduction after the pooling it is denoted with p. 1x1, 3x3, and 5x5 pooling are then
concatenated to get the final output
'''

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, f1x1, f3x3red, f3x3, f3x3_branches, f5x5red, f5x5, pool_proj, pool_type='max'):
        super(InceptionModule, self).__init__()
        
        # Branch 1
        self.branch1 = ConvBlock(
            in_channels  = in_channels, 
            out_channels = f1x1, 
            kernel_size  = 1, 
            stride       = 1, 
            padding      = 0
        ) if f1x1 > 0 else None
        
        # Branch 2
        self.branch2 = nn.ModuleList()
        for _ in range(f3x3_branches):
            branch = nn.Sequential(
                ConvBlock(
                    in_channels  = in_channels, 
                    out_channels = f3x3red, 
                    kernel_size  = 1, 
                    stride       = 1, 
                    padding      = 0
                ),

                ConvBlock(
                    in_channels  = f3x3red, 
                    out_channels = f3x3, 
                    kernel_size  = 3, 
                    stride       = 1, 
                    padding      = 1
                )
            ) if f3x3 > 0 else None
            self.branch2.append(branch)
        
        # Branch 3
        self.branch3 = nn.ModuleList()
        for _ in range(f3x3_branches):
            branch = nn.Sequential(
                ConvBlock(
                    in_channels  = in_channels,
                    out_channels = f5x5red,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0
                ),
                ConvBlock(
                    in_channels  = f5x5red,
                    out_channels = f5x5,
                    kernel_size  = 5,
                    stride       = 1,
                    padding      = 2
                )
            ) if f5x5 > 0 else None
            self.branch3.append(branch)
        
        # Branch 4
        if pool_proj > 0:
            if pool_type == 'max':
                self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    
                    ConvBlock(
                        in_channels  = in_channels, 
                        out_channels = pool_proj, 
                        kernel_size  = 1, 
                        stride       = 1, 
                        padding      = 0
                    )
                )
            
            elif pool_type == 'L2':
                self.branch4 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                    
                    ConvBlock(
                        in_channels  = in_channels, 
                        out_channels = pool_proj, 
                        kernel_size  = 1, 
                        stride       = 1, 
                        padding      = 0
                    ),
                    
                    L2Norm()
                )
        else:
            self.branch4 = None
    
    def forward(self, x):
        branches = [b(x) for b in [self.branch1] + list(self.branch2) + list(self.branch3) + [self.branch4] if b is not None]
        return torch.cat(branches, 1)
    
class FaceNet(nn.Module):
    def __init__(self, emb_size: int = 64, restore_from_checkpoint: str = None):
        super(FaceNet, self).__init__()
        
        if restore_from_checkpoint is not None:
            try:
                load = torch.load(restore_from_checkpoint)
                load = {k.replace('_orig_mod.', ''): v for k, v in load.items()}
                self.load_state_dict(load)
            except FileNotFoundError:
                print(f"Checkpoint '{restore_from_checkpoint}' not found.")
            except RuntimeError as e:
                print(f"Error loading the checkpoint: {e}")
        
        self.conv1 = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        
        self.max_pool_norm = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(64)
        )
        
        self.inception2_1 = InceptionModule(in_channels=64, f1x1=0, f3x3red=192, f3x3=192, f3x3_branches=1, f5x5red=0, f5x5=0, pool_proj=0, pool_type='max')
        self.inception2_2 = InceptionModule(in_channels=192, f1x1=0, f3x3red=192, f3x3=192, f3x3_branches=1, f5x5red=0, f5x5=0, pool_proj=0, pool_type='max')
        
        self.norm = nn.LocalResponseNorm(192)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = InceptionModule(in_channels=192, f1x1=64, f3x3red=96, f3x3=128, f3x3_branches=1, f5x5red=16, f5x5=32, pool_proj=32, pool_type='max')
        self.inception3b = InceptionModule(in_channels=256, f1x1=64, f3x3red=96, f3x3=128, f3x3_branches=1, f5x5red=32, f5x5=64, pool_proj=64, pool_type='L2')
        self.inception3c = InceptionModule(in_channels=320, f1x1=0, f3x3red=128, f3x3=256, f3x3_branches=2, f5x5red=32, f5x5=64, pool_proj=0, pool_type='max')
        
        self.inception4a = InceptionModule(in_channels=640, f1x1=256, f3x3red=96, f3x3=192, f3x3_branches=1, f5x5red=32, f5x5=64, pool_proj=128, pool_type='L2')
        self.inception4b = InceptionModule(in_channels=640, f1x1=224, f3x3red=112, f3x3=224, f3x3_branches=1, f5x5red=32, f5x5=64, pool_proj=128, pool_type='L2')
        self.inception4c = InceptionModule(in_channels=640, f1x1=192, f3x3red=128, f3x3=256, f3x3_branches=1, f5x5red=32, f5x5=64, pool_proj=128, pool_type='L2')
        self.inception4d = InceptionModule(in_channels=640, f1x1=160, f3x3red=144, f3x3=288, f3x3_branches=1, f5x5red=32, f5x5=64, pool_proj=128, pool_type='L2')
        self.inception4e = InceptionModule(in_channels=640, f1x1=0, f3x3red=160, f3x3=256, f3x3_branches=2, f5x5red=64, f5x5=128, pool_proj=256, pool_type='max')
        
        self.inception5a = InceptionModule(in_channels=1024, f1x1=384, f3x3red=192, f3x3=384, f3x3_branches=1, f5x5red=48, f5x5=128, pool_proj=128, pool_type='L2')
        self.inception5b = InceptionModule(in_channels=1024, f1x1=384, f3x3red=192, f3x3=384, f3x3_branches=1, f5x5red=48, f5x5=128, pool_proj=128, pool_type='max')
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(1024, emb_size)
        
        self.l2_norm = L2Norm()
        self.droupout = nn.Dropout(0.4)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool_norm(x)

        x = self.inception2_1(x)
        x = self.inception2_2(x)
        
        x = self.norm(x)
        x = self.max_pool(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool(self.inception3c(x))

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.max_pool(self.inception4e(x))
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.droupout(x)
        
        x = self.fc(x)
        x = self.l2_norm(x)
        return x