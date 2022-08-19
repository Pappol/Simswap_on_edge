from torchinfo import summary
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions

def main(opt):
    torch.nn.Module.dump_patches = True
    model = create_model(opt)

    summary(model, [(1, 3, 224, 224), (1, 3, 224, 224), (1, 512), (1, 512)], dtypes=[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool])

if __name__ == '__main__':
    opt = TestOptions().parse()

    main(opt)