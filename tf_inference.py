import numpy as np
import os
import argparse
import tflite_runtime.interpreter as tflite
from matplotlib import pyplot as plt
import time
from PIL import Image
from torchvision import transforms
import torch
import cv2
from torchvision import transforms as T

def long_benchmark(args):
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    #inport and model data
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    pic_a = args.pic_a_path
    img_a = c_transforms(Image.open(pic_a))


    src_image1  = img_a.sub_(imagenet_mean).div_(imagenet_std)

    src_image1 = src_image1.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    #create interpreter
    interpreter = tflite.Interpreter(args.model_path, num_threads=24)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #import latent id
    latent_id = torch.load('output/latent_id.pth')

    #inference
    start = time.time()
    interpreter.set_tensor(input_details[1]['index'],latent_id.numpy())
    interpreter.set_tensor(input_details[0]['index'], src_image1.numpy())
    times = []
    for i in range(11):
        start = time.time()    
        interpreter.invoke()
        end = time.time()
        times.append(end - start)
        print("time: ", end - start)
        print("\n")

    print("average time: ", sum(times[1:])/10)

def postprocess(x):
    """[0,1] to uint8."""
    x = np.clip(255 * x, 0, 255)
    x = np.cast[np.uint8](x)
    return x



def benchmark(args):
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    #inport and model data
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    pic_a = args.pic_a_path
    img_a = c_transforms(Image.open(pic_a))


    src_image1  = img_a.sub_(imagenet_mean).div_(imagenet_std)

    src_image1 = src_image1.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    #create interpreter
    interpreter = tflite.Interpreter(args.model_path, num_threads=24)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #import latent id
    latent_id = torch.load('output/latent_id.pth')

    #inference
    start = time.time()
    interpreter.set_tensor(input_details[1]['index'],latent_id.numpy())
    interpreter.set_tensor(input_details[0]['index'], src_image1.numpy())
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    print("time: ", end - start)
    print("\n")
    
    #save result
    result = output_data[0]
    #convert result to image
    img_fake    = result * imagenet_std.cpu().detach().numpy()

    img_fake    = img_fake + imagenet_mean.cpu().detach().numpy()
    result = np.transpose(img_fake, (1, 2, 0))
    result = result * 255
    result = result.astype(np.uint8)
    result = Image.fromarray(result)
    result.save('output/result.png')

def main(args):
    if args.benchmark == 'long':
        long_benchmark(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/model.tflite",
                            help="path of onnx extra data folder"),
    parser.add_argument("--pic_a_path", type=str, default="./crop_224/mtdm.jpg",
                            help="path of image a"),
    parser.add_argument("--output_path", type=str, default="output/",
                            help="path of output folder"),
    parser.add_argument("--benchmark", type=str, default="short",   
                            help="benchmark type, long or short"),
    args = parser.parse_args()

    main(args)
