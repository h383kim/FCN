'''
This is the implementation of FCN_8s 
using vgg16_bn backbone pretrained on ImageNet
'''
import torch
from torch import nn
import torchvision.models as models

class FCN_8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        vgg16 = models.vgg16_bn(weights="IMAGENET1K_V1", progress=True)

        # Use the features from vgg16
        self.features = vgg16.features

        # Replace the classifier with convolutional layers
        self.score_pool4 = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.score_pool3 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.score_fr = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Transposed convolution layers for upsampling
        '''
        score_fr*2 means score_fr upsampled by factor of 2 using Transposed Convolution
        '''
        self.upscore_pool5 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        ) # Upsamples the score_fr by factor of 2
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        ) # Upsamples the (score_fr*2 + score_pool4) by factor of 2
        self.upscore_pool3 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4
        ) # Upsamples the [(score_fr*2 + score_pool4)*2 + score_pool3] by factor of 8

    def forward(self, x):
        # Store intermediate outputs for skip connections
        pool3 = None
        pool4 = None

        '''input x: [N, 3, 384, 480] '''
        # Forward pass through VGG16 features
        for i in range(len(self.features)):
            x = self.features[i](x) # Feed forwarding the previous output to each layer coming next
            if i == 23: # After pool_3 layer passed
                pool3 = x 
                ''' pool3: [N, 256, 48, 60] '''
            elif i == 33: # After pool_4 layer passed
                pool4 = x 
                ''' pool4: [N, 512, 24, 30] '''
            elif i == 43: 
                ''' x: [N, 512, 12, 15] '''
                break;

        # Classify the features
        # x is now the output from the last pooling layer(pool_5) of vgg16_bn
        ''' score_pool5: [N, 32, 12, 15] '''
        score_pool5 = self.score_fr(x) # (N, num_classes, H/32, W/32)
        # Upsample the pool5 score by factor of 2
        ''' score_pool5: [N, 32, 24, 30] '''
        score_pool5 = self.upscore_pool5(score_pool5) # (N, num_classes, H/16, W/16)
        
        # Add skip connection from pool4_out
        ''' score_pool4: [N, 32, 24, 30] '''
        score_pool4 = self.score_pool4(pool4)
        score_pool4_5 = score_pool5 + score_pool4

        # Upsample the skip-connected pool4+pool5 score by factor of 2
        ''' score_pool4_5: [N, 32, 48, 60] '''
        score_pool4_5 = self.upscore_pool4(score_pool4_5) # (N, num_classes, H/8, W/8)
        
        # Add skip connection from pool3
        ''' score_pool3: [N, 32, 48, 60] '''
        score_pool3 = self.score_pool3(pool3)
        score_final = score_pool4_5 + score_pool3

        # Finally, upsample the skip-connected pool3+pool4+pool5 score by factor of 8
        score_final = self.upscore_pool3(score_final) # (N, num_classes, H, W)
        ''' score_final: [N, 32, 384, 480] '''
        # The output tensor now has the same spatial dimensions as the input
        return score_final