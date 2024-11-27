'''
This is the implementation of FCN_8s
using baseline(non-trained) VGG16 backbone 
'''

import torch
from torch import nn
import torchvision.models as models

class VGG16_FCN(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        
        # Use the features from vgg16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, C, H/2, W/2] = [N, 64, 192, 240]
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, C, H/4, W/4] = [N, 128, 96, 120]
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, C, H/8, W/8] = [N, 256, 48, 60]
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, C, H/16, W/16] = [N, 512, 24, 30]
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [N, C, H/32, W/32] = [N, 512, 12, 15]
        )

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
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        '''
        Transposed convolution layers for upsampling.
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
        ''' input x: [N, 3, 384, 480] '''
        pool1_out = self.conv1(x)
        ''' pool1_out: [N, 64, 192, 240] '''
        pool2_out = self.conv2(pool1_out)
        ''' pool2_out: [N, 128, 96, 120] '''
        pool3_out = self.conv3(pool2_out)
        ''' pool3_out: [N, 256, 48, 60] '''
        pool4_out = self.conv4(pool3_out)
        ''' pool4_out: [N, 512, 24, 30] '''
        pool5_out = self.conv5(pool4_out)
        ''' pool5_out: [N, 512, 12, 15] '''


        # Classify the features
        ''' score_pool5: [N, 32, 12, 15] '''
        score_pool5 = self.score_fr(pool5_out) # (N, num_classes, H/32, W/32)
        # Upsample the pool5 score by factor of 2
        ''' score_pool5: [N, 32, 24, 30] '''
        score_pool5 = self.upscore_pool5(score_pool5) # (N, num_classes, H/16, W/16)
        
        # Add skip connection from pool4_out
        ''' score_pool4: [N, 32, 24, 30] '''
        score_pool4 = self.score_pool4(pool4_out)
        score_pool4_5 = score_pool5 + score_pool4

        # Upsample the skip-connected pool4+pool5 score by factor of 2
        ''' score_pool4_5: [N, 32, 48, 60] '''
        score_pool4_5 = self.upscore_pool4(score_pool4_5) # (N, num_classes, H/8, W/8)
        
        # Add skip connection from pool3
        ''' score_pool3: [N, 32, 48, 60] '''
        score_pool3 = self.score_pool3(pool3_out)
        score_final = score_pool4_5 + score_pool3

        # Finally, upsample the skip-connected pool3+pool4+pool5 score by factor of 8
        score_final = self.upscore_pool3(score_final) # (N, num_classes, H, W)
        ''' score_final: [N, 32, 384, 480] '''
        # The output tensor now has the same spatial dimensions as the input
        return score_final
        