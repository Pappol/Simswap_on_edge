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


def test_latent_id(args):
    target =cv2.imread(args.pic_a_path).astype(np.float32)
    target = cv2.resize(target, (224, 224))
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    target = (np.transpose(target, (2,0,1)))/255.0
    target = target[np.newaxis,:]

    interpreter = tflite.Interpreter(args.model_path, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latent_id = torch.load('output/latent_id.pth')
    #convert to numpy
    latent_id = latent_id.numpy()
    #print dimensions of latent_id
    print ("Latent id shape: ", latent_id.shape)
    #print input details dimensions
    for i in input_details:
        print(i['shape'])


def long_benchmark(args):
    #load data
    transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    pic_a = args.pic_a_path
    img_a = Image.open(pic_a).convert('RGB')
    img_a = transformer_Arcface(img_a)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    #create interpreter
    interpreter = tflite.Interpreter(args.model_path, num_threads=24)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #import latent id
    latent_id = torch.load('output/latent_id.pth')

    #inference
    for i in range(10):
        start = time.time()
        interpreter.set_tensor(input_details[1]['index'],latent_id.numpy())
        interpreter.set_tensor(input_details[0]['index'], img_id.numpy())
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end = time.time()
        print("time: ", end - start)
        print("\n")
    

    
def tester(args):
    target =cv2.imread(args.pic_a_path).astype(np.float32)
    target = cv2.resize(target, (224, 224))
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    target = (np.transpose(target, (2,0,1)))/255.0
    target = target[np.newaxis,:]

    interpreter = tflite.Interpreter(args.model_path, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latent_id = torch.load('output/latent_id.pth')
    #convert to numpy
    latent_id = latent_id.numpy()

    #inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'],latent_id.numpy())
    interpreter.set_tensor(input_details[1]['index'], target)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    print("time: ", end - start)
    print("\n")
    
    #save result
    image =output_data[0]
    print (image.shape)
    image = (image*255.0).transpose(1,2,0).astype(np.uint8)[:,:,::-1]
    print (image.shape)
    cv2.imshow('I hate onix', image)
    cv2.imwrite('output/out_onnx.png', image)
    cv2.waitKey()

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
    #test_latent_id(args)
    #tester(args)
    benchmark(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/model.tflite",
                            help="path of onnx extra data folder"),
    parser.add_argument("--pic_a_path", type=str, default="./crop_224/mtdm.jpg",
                            help="path of image a"),
    parser.add_argument("--output_path", type=str, default="output/",
                            help="path of output folder"),

    args = parser.parse_args()

    main(args)
