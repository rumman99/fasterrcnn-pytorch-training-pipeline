import torchvision
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead

'''
fasterrcnn_resnet50_fpn_v2 - Default
====================================================================================================
Total params: 43,712,278
Trainable params: 43,486,934
Non-trainable params: 225,344
Total mult-adds (T): 1.12
====================================================================================================
Input size (MB): 19.66
Forward/backward pass size (MB): 14970.78
Params size (MB): 174.85
Estimated Total Size (MB): 15165.29
====================================================================================================


fasterrcnn_resnet50_fpn_v2 - custom 1
1) Added a single block to the bottle neck
2) Added one extra block to RPN

====================================================================================================
Total params: 46,611,940
Trainable params: 46,386,596
Non-trainable params: 225,344
Total mult-adds (T): 1.62
====================================================================================================
Input size (MB): 19.66
Forward/backward pass size (MB): 17371.85
Params size (MB): 186.45
Estimated Total Size (MB): 17577.96
====================================================================================================
'''



class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dNormActivation, self).__init__()
        self.add_module("0", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.add_module("1", nn.BatchNorm2d(out_channels))
        self.add_module("2", nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        x = self._modules["2"](x)
        return x
    
class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(RegionProposalNetwork, self).__init__()
        self.add_module("0", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        self.add_module("1", nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        return x


class add_CustomBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(add_CustomBottleNeck, self).__init__()
        self.add_module("conv1", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.add_module("bn1", nn.BatchNorm2d(out_channels))
        self.add_module("conv2", nn.Conv2d(out_channels, 256, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.add_module("bn2", nn.BatchNorm2d(256))
        self.add_module("relu", nn.ReLU(inplace=True))
    
    def forward(self, x):
        x = self._modules["conv1"](x)
        x = self._modules["bn1"](x)
        x = self._modules["conv2"](x)
        x = self._modules["bn2"](x)
        x = self._modules["relu"](x)
        return x
        


def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights='DEFAULT'
    )
    
    # print(model.backbone.body.layer1)
    model.backbone.body.layer1.insert(3, add_CustomBottleNeck(256, 512))

    model.rpn.head.conv.insert(2, RegionProposalNetwork(256, 256))
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model, coco_model
        
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=False)
    summary(model)

