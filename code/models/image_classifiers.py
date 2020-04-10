__author__ = 'marvinler'

import torch.nn as nn
import torchvision.models as tm
from torchvision.models.alexnet import model_urls as alexnet_models_urls
from torchvision.models.densenet import model_urls as densenet_models_urls
from torchvision.models.googlenet import model_urls as googlenet_models_urls
from torchvision.models.inception import model_urls as inception_models_urls
from torchvision.models.mnasnet import _MODEL_URLS as mnasnet_models_urls
from torchvision.models.mobilenet import model_urls as mobilenet_models_urls
from torchvision.models.resnet import model_urls as resnet_models_urls
from torchvision.models.shufflenetv2 import model_urls as shufflenetv2_models_urls
from torchvision.models.squeezenet import model_urls as squeezenet_models_urls
from torchvision.models.vgg import model_urls as vgg_models_urls

ALL_AVAILABLE_MODELS = [model_url for models_urls in
                        [alexnet_models_urls, densenet_models_urls, googlenet_models_urls, inception_models_urls,
                         mnasnet_models_urls, mobilenet_models_urls, resnet_models_urls, shufflenetv2_models_urls,
                         squeezenet_models_urls, vgg_models_urls] for model_url in models_urls.keys()]


def get_original_classifier(model_type, pretrained):
    assert model_type.lower() in ALL_AVAILABLE_MODELS, \
        'Requested model type "%s" not in available models: %s' % (model_type.lower(), str(ALL_AVAILABLE_MODELS))

    model = getattr(tm, model_type.lower())(pretrained, progress=True)
    return model


def replace_last_layer(model, model_type, n_classes):
    if 'resnet' in model_type or 'googlenet' in model_type or 'inception' in model_type or 'shufflenetv2' in model_type:
        model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)
    elif 'densenet' in model_type:
        model.classifier = nn.Linear(model.classifier.in_features, n_classes, bias=True)
    elif 'mnasnet' in model_type or 'mobilenet' in model_type:
        model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features, n_classes, bias=True)
    elif 'squeezenet' in model_type:
        model.classifier._modules['1'] = nn.Conv2d(model.classifier._modules['1'].in_channels, n_classes, kernel_size=1)
    elif 'vgg' in model_type:
        model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, n_classes, bias=True)
    else:
        raise ValueError('Unknow type %s' % model_type)

    return model


def instantiate_model(model_type, pretrained, n_classes):
    model = get_original_classifier(model_type, pretrained)
    model = replace_last_layer(model, model_type, n_classes)

    input_width = 299 if 'inception' in model_type else 224
    return model, input_width


if __name__ == '__main__':
    model = instantiate_model('vgg19', True, 1)
    print(model)
