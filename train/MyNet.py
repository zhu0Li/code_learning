import math

from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Yolo_v1(nn.Module):
    def __init__(self, num_class):
        super(Yolo_v1, self).__init__()
        C = num_class
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=7 // 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 为了简便，这里省去了很多层
        self.flatten = Flatten()
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            nn.Dropout(0.5), nn.LeakyReLU(0.1))
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + C)))
        self._initialize_weights()


    def forward(self, input):
        conv_layer1 = self.conv_layer1(input)
        conv_layer2 = self.conv_layer2(conv_layer1)
        flatten = self.flatten(conv_layer2)
        conn_layer1 = self.conn_layer1(flatten)
        output = self.conn_layer2(conn_layer1)
        return output


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

## 构建很多层
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None):
      super(BasicBlock, self).__init__()
      self.conv1 = conv3x3(inplanes, planes, stride)
      self.bn1 = norm_layer(planes)
      self.relu = nn.ReLU(inplace=True)
      self.conv2 = conv3x3(planes, planes)
      self.bn2 = norm_layer(planes)
      self.downsample = downsample
      self.stride = stride

    def forward(self, x):
      identity = x
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)
      out = self.conv2(out)
      out = self.bn2(out)
      if self.downsample is not None:
          identity = self.downsample(x)
      out += identity
      out = self.relu(out)
      return out
    
layers = []
layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
self.base_width, previous_dilation, norm_layer))
self.inplanes = planes * block.expansion
for _ in range(1, blocks):
    layers.append(block(self.inplanes, planes, groups=self.groups,
    base_width=self.base_width, dilation=self.dilation,
    norm_layer=norm_layer))
return nn.Sequential(*layers)