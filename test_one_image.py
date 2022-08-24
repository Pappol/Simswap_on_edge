
from cgi import test
from copyreg import pickle
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
import time
import os
import PIL
from PIL import Image 
from models.projected_model import fsModel
from torchvision import transforms as T
from data.data_loader_Swapping import GetLoader
from util.plot import plot_batch

import glob

def benchmark(model, path):
    #select list of images from folder
    list_of_images = glob.glob(path, recursive=False)
    print(list_of_images)
    model.eval()
    for i in range(list_of_images.__len__()-1):
        start_time = time.time()

        pic_a = list_of_images[i]
        pic_b = list_of_images[i+1]
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])
        # convert numpy to tensor
        img_id = img_id.cuda()
        img_att = img_att.cuda()
        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)

        latend_id = latend_id.to('cuda')
        #forward
        img_fake = model(img_id, img_att, latend_id, latend_id, True)\
        
        end_time = time.time()
        print('Time: %f s' % (end_time - start_time))
        #save image
        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]

        output = output*255

        cv2.imwrite(opt.output_path + 'result '+str(i)+'.jpg',output)



def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

def print_model_parameters(model, opt):

    model.eval()
    pic_a = opt.pic_a_path
    img_a = Image.open(pic_a).convert('RGB')
    img_a = transformer_Arcface(img_a)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    pic_b = opt.pic_b_path

    img_b = Image.open(pic_b).convert('RGB')
    img_b = transformer(img_b)
    img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

    # convert numpy to tensor
    img_id = img_id.cuda()
    img_att = img_att.cuda()

    #create latent id
    img_id_downsample = F.interpolate(img_id, size=(112,112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
    print("-------------Type--------------")
    print(type(img_id))
    print(type(img_att))
    print (type(latend_id))
    print (type(latend_id))
    print (type(True))
    print("-------------Shape--------------")
    print(img_id.shape)
    print(img_att.shape)
    print(latend_id.shape)
    print(latend_id.shape)


def postprocess(x):
    """[0,1] to uint8."""
    x = np.clip(255 * x, 0, 255)
    x = np.cast[np.uint8](x)
    return x


def old_test_one_image(opt):

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        
        pic_a = opt.pic_a_path
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        pic_b = opt.pic_b_path

        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
        latend_id = latend_id.to('cuda')


        ############## Forward Pass ######################
        img_fake = model(img_id, img_att, latend_id, latend_id, True)

        full = img_fake[0].detach()
        imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).cuda()
        imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).cuda()
        full = full * imagenet_std + imagenet_mean

        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = postprocess(output)
        output = output*255
        

        cv2.imwrite(opt.output_path + 'result.jpg',output)

def test_one_image(opt):
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).cuda()
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).cuda()
    #inport and model data
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    pic_a = opt.pic_a_path
    img_a = c_transforms(Image.open(pic_a))
    pic_b = opt.pic_b_path
    img_b = c_transforms(Image.open(pic_b))

    src_image1  = img_a.cuda(non_blocking=True)
    src_image1  = src_image1.sub_(imagenet_mean).div_(imagenet_std)
    src_image2  = img_b.cuda(non_blocking=True)
    src_image2  = src_image2.sub_(imagenet_mean).div_(imagenet_std)


    src_image1 = src_image1.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    imagenet_std   = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean  = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    model = fsModel()

    model.initialize(opt)
    model.netG.eval()

    with torch.no_grad():
        arcface_112     = F.interpolate(src_image1,size=(112,112))
        id_vector_src1  = model.netArc(arcface_112)
        id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

        img_fake    = model.netG(src_image2, id_vector_src1).cpu()
                    
        img_fake    = img_fake * imagenet_std
        img_fake    = img_fake + imagenet_mean
        img_fake    = img_fake.numpy()
        img_fake    = img_fake.transpose(0,2,3,1)
        img_fake    = img_fake[0]
        img_fake    = postprocess(img_fake)
        img_fake    = img_fake*255
        cv2.imwrite(opt.output_path + 'result.jpg',img_fake)




def main(opt):
    #benchmark(model, "./crop_224/*.jpg")
    test_one_image(opt)

if __name__ == '__main__':
    opt = TestOptions().parse()

    main(opt)