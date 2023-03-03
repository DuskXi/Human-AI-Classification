import torchvision.models as models
from torch import nn


class ModelLoader:
    def __init__(self):
        self.available_models = {
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "efficientnet-b0": models.efficientnet_b0,
            "efficientnet-b1": models.efficientnet_b1,
            "efficientnet-b2": models.efficientnet_b2,
            "efficientnet-b3": models.efficientnet_b3,
            "efficientnet-b4": models.efficientnet_b4,
            "efficientnet-b5": models.efficientnet_b5,
            "efficientnet-b6": models.efficientnet_b6,
            "efficientnet-b7": models.efficientnet_b7,
            "efficientnet_v2_s": models.efficientnet_v2_s,
            "efficientnet_v2_m": models.efficientnet_v2_m,
            "efficientnet_v2_l": models.efficientnet_v2_l,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "googlenet": models.googlenet,
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
        }

    def create_model(self, model_name="vgg16", pretrained=True, num_classes=2):
        if model_name.startswith("vgg"):
            model = self.available_models[model_name](pretrained=pretrained)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )
        elif model_name.startswith("resnet") or model_name.startswith("inception") or model_name.startswith("googlenet"):
            model = self.available_models[model_name](pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name.startswith("densenet"):
            model = self.available_models[model_name](pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name.startswith("efficientnet"):
            model = self.available_models[model_name](pretrained=pretrained)
            model.classifier = nn.Sequential(
                nn.Dropout(p=model.classifier[0].p, inplace=True),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
            model.classifier.modules()
        else:
            model = self.available_models[model_name](pretrained=pretrained, num_classes=num_classes)
        return model
