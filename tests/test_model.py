import torch

from galaxy_vit.models.vit import GalaxyViT


def test_model_output_shape():
    model = GalaxyViT('vit_tiny_patch16_224', num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
