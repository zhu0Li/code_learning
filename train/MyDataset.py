import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class MyDataset(data.Dataset):
   def __init__(self,data_folder):
       '''
       init函数主要是完成两个静态变量的赋值。一个是用于存储所有数据路径的变量，变量的每个元素即为一份训练样本，
       （注：如果一份样本是十几帧图像，则变量每个元素存储的是这十几帧图像的路径），可以命名为self.filenames。
       一个是用于存储与数据路径变量一一对应的标签变量，可以命名为self.labels。
       '''
       self.data_folder = data_folder
       self.filenames = []
       self.labels = []

       per_classes = os.listdir(data_folder)
       for per_class in per_classes:
           per_class_paths = os.path.join(data_folder, per_class)
           label = torch.tensor(int(per_class))

           per_datas = os.listdir(per_class_paths)
           for per_data in per_datas:
               self.filenames.append(os.path.join(per_class_paths, per_data))
               self.labels.append(label)

   def __getitem__(self, index):
       '''
       getitem 函数主要是根据索引返回对应的数据。这个索引是在训练前通过dataloader切片获得的。
       它的参数默认是index，即每次传回在init函数中获得的所有样本中索引对应的数据和标签。
       '''
       image = Image.open(self.filenames[index])
       label = self.labels[index]
       data = self.proprecess(image)
       return data, label

   def __len__(self):
       '''
       len函数主要就是返回数据长度，即样本的总数量。前面介绍了self.filenames的每个元素即为每份样本的路径，
       因此，self.filename的长度就是样本的数量。通过return len(self.filenames)即可返回数据长度。
       '''
       return len(self.filenames)

   def proprecess(self,data):
       transform_train_list = [
           transforms.Resize((self.opt.h, self.opt.w), interpolation=3),
           transforms.Pad(self.opt.pad, padding_mode='edge'),
           transforms.RandomCrop((self.opt.h, self.opt.w)),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]
       return transforms.Compose(transform_train_list)(data)