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

    pic_b = args.pic_b_path

    img_b = Image.open(pic_b).convert('RGB')
    img_b = transformer(img_b)
    img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

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
        interpreter.set_tensor(input_details[0]['index'],latent_id.numpy())
        interpreter.set_tensor(input_details[1]['index'], img_id.numpy())
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end = time.time()
        print("time: ", end - start)
        print("\n")
    
def benchmark(args):
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

    pic_b = args.pic_b_path

    img_b = Image.open(pic_b).convert('RGB')
    img_b = transformer(img_b)
    img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

    #create interpreter
    interpreter = tflite.Interpreter(args.model_path, num_threads=24)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in output_details:
        print(i['shape'])
        print("\n")
    print("\n")


    for i in input_details:
        print(i['shape'])
    print("\n")

    #import latent id
    latent_id = torch.load('output/latent_id.pth')

    #inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'],latent_id.numpy())
    interpreter.set_tensor(input_details[1]['index'], img_id.numpy())
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    print("time: ", end - start)
    print("\n")
    
    #save result
    result = output_data[0]
    #convert result to image
    result = np.transpose(result, (1, 2, 0))
    result = result * 255
    result = result.astype(np.uint8)
    result = Image.fromarray(result)
    result.save('output/result.png')

    for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = output_data[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, output_data[i]], dim=2)

    full = row3
    full = np.transpose(full, (1, 2, 0))
    output = full
    output = np.array(output)
    output = output[..., ::-1]

    output = output*255

    cv2.imwrite(args.output_path + 'result_tf.jpg',output)

def main(args):
    long_benchmark(args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/model.tflite",
                            help="path of onnx extra data folder"),
    parser.add_argument("--pic_a_path", type=str, default="./crop_224/gdg.jpg",
                            help="path of image a"),
    parser.add_argument("--pic_b_path", type=str, default="./crop_224/zrf.jpg",
                            help="path of image b"),
    parser.add_argument("--z_id_path", type=str, default="preprocess/z_id.npy",
                            help="path of z_id tensor"),
    parser.add_argument("--output_path", type=str, default="output/",
                            help="path of output folder"),
    parser.add_argument("--target_image", type=str, default="data/faceshifter-datasets-preprocessed/train/00000056.png",
                            help="path of preprocessed target face image"),

    args = parser.parse_args()

    main(args)

    """
    
    img = cv2.imread(args.target_image).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.transpose(2,0,1)/255.0
    img = img[np.newaxis, :]

    z_id = np.load(args.z_id_path).astype(np.float32)

    interpreter = tflite.Interpreter(args.model_path+ "MultiLevelEncoder_gen_Lite_optimized.tflite", num_threads=10)
    interpreter.allocate_tensors()
    interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=10)
    interpreter_ADD.allocate_tensors()

    interpreter_ADD = tflite.Interpreter(args.model_path+ "ADD_gen_Lite_optimized.tflite", num_threads=10)
    interpreter_ADD.allocate_tensors()

    input_details_ADD = interpreter_ADD.get_input_details()
    output_details_ADD = interpreter_ADD.get_output_details()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i in range(0, 10):
        start_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        feature_map = []

        for i in range(0,7):
            feature_map.append(interpreter.get_tensor(output_details[i]['index']))

        print("multi --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        for i in range(0,5):
            interpreter_ADD.set_tensor(input_details_ADD[i]['index'], feature_map[i])

        interpreter_ADD.set_tensor(input_details_ADD[6]['index'], z_id)

        interpreter_ADD.set_tensor(input_details_ADD[7]['index'], feature_map[6])

        interpreter_ADD.invoke()
        print("ADD --- %s seconds ---" % (time.time() - start_time))
    """
