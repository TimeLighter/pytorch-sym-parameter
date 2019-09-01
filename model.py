import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
norm_layer = nn.InstanceNorm2d

class GateLayer(nn.Module): #avg pool
   def __init__(self, channel, cond_size, reduction=4):
       super(GateLayer, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.fc = nn.Sequential(
               nn.Linear(channel*2, channel // reduction),
               nn.ReLU(inplace=True),
               nn.Linear(channel // reduction, channel),
               nn.Sigmoid()
       )

       self.l_c = nn.Linear(cond_size, channel)
       self.channel = channel

   def forward(self, x, cond):

       b, c, _, _ = x.size()
       y = self.avg_pool(x).view(b, c)

       cond = F.relu(self.l_c(cond))
       y = torch.cat([y, cond], 1)

       y = self.fc(y).view(b, self.channel, 1, 1)
       return y



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        # nn.InstanceNorm2d(in_features),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        # nn.InstanceNorm2d(in_features)
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)





class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, cond_size=3):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    # nn.InstanceNorm2d(64),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)
        self.gate0 = GateLayer(64, cond_size)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        # nn.InstanceNorm2d(out_features),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)
        self.gate1 = GateLayer(in_features, cond_size)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)
        self.gate2 = GateLayer(in_features, cond_size)


        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        # nn.InstanceNorm2d(out_features),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)
        self.gate3 = GateLayer(64, cond_size)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond):
        out = self.model0(x)
        out = self.gate0(out, cond) * out

        out = self.model1(out)
        out = self.gate1(out, cond) * out

        out = self.model2(out)
        out = self.gate2(out, cond) * out

        out = self.model3(out)
        # out = self.gate3(out, cond) * out

        out = self.model4(out)

        return out


