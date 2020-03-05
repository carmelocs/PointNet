import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This is Spatial Transform Network to guarantee the invariance of rigid transform (rotation and translation).
class STNkd(nn.Module):
  def __init__(self, k=3):
    super(STNkd, self).__init__()
    self.mlp = nn.Sequential(
        nn.Conv1d(k, 64, 1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 128, 1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Conv1d(128, 1024, 1),
        nn.BatchNorm1d(1024),
        nn.ReLU()
    )

    self.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, k*k)
    )
    self.k = k
    

  def forward(self, x):
      batch_size = x.size()[0]
      #print('input size: ', x.shape)
      x = self.mlp(x)
      #print('size after mlp: ', x.shape)
      x = x.max(2)[0]
      #print('size after maxpooling: ', x.shape)
      x = self.fc(x)
      #print('size after fully connection: ', x.shape)

      iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k*self.k).repeat(batch_size, 1)
      if x.is_cuda:
        iden = iden.cuda()
      x = x + iden
      x = x.view(-1, self.k, self.k)
      #print('STN size: ', x.shape)

      return x

class PointNetFeat(nn.Module):
  def __init__(self, global_feature=True, feature_transform=False):
    super(PointNetFeat, self).__init__()
    self.stn3d = STNkd(k=3)
    self.conv1 = nn.Conv1d(3, 64, 1)
    self.conv2 = nn.Conv1d(64, 64, 1)
    self.conv3 = nn.Conv1d(64, 64, 1)
    self.conv4 = nn.Conv1d(64, 128, 1)
    self.conv5 = nn.Conv1d(128, 1024, 1)
    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(64)
    self.bn4 = nn.BatchNorm1d(128)
    self.bn5 = nn.BatchNorm1d(1024)
    self.global_feature = global_feature
    self.feature_transform = feature_transform
    if feature_transform:
      self.fstn = STNkd(k=64)

  def forward(self, x):
    num_points = x.size()[2]
    trans = self.stn3d(x)
    x = x.transpose(2,1)
    x = torch.bmm(x, trans)
    x = x.transpose(2,1)
    #print('size after input transform: ', x.shape)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    
    if self.feature_transform:
      feat_trans = self.fstn(x)
      x = x.transpose(2,1)
      x = torch.bmm(x, feat_trans)
      x = x.transpose(2,1)

    local_feature = x
    #print('local feature size: ', local_feature.shape)
    x = F.relu(self.bn4(self.conv4(x)))
    x = F.relu(self.bn5(self.conv5(x)))
    x = x.max(2)[0]
    #print('size after global maxpooling: ', x.shape)

    if self.global_feature:
      return x
    else:
      x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
      return torch.cat((x, local_feature), 1)

class PointNetCls(nn.Module):
  def __init__(self, num_classes, feature_transform=False):
    super(PointNetCls, self).__init__()
    self.feature = PointNetFeat(feature_transform=feature_transform)
    self.classifer = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.Dropout(0.3),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

  def forward(self, x):
    x = self.feature(x)
    x = self.classifer(x)
    print('classification output size: ', x.shape)

    return F.log_softmax(x, dim=-1)

class PointNetSeg(nn.Module):
  def __init__(self, num_parts, feature_transform=False):
    super(PointNetSeg, self).__init__()
    self.num_parts = num_parts
    self.feature_transform = feature_transform
    self.feature = PointNetFeat(global_feature=False, feature_transform=self.feature_transform)
    self.mlp = nn.Sequential(
        nn.Conv1d(1088, 512, 1),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Conv1d(512, 256, 1),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Conv1d(256, 128, 1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Conv1d(128, 128, 1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Conv1d(128, self.num_parts, 1)
    )

  def forward(self, x):
    batch_size = x.size()[0]
    num_points = x.size()[2]
    feature = self.feature(x)
    x = self.mlp(feature)
    x = F.log_softmax(x.view(batch_size, num_points, self.num_parts),-1)
    print('segmentation output size: ', x.shape)

    return x

if __name__ == '__main__':
    INPUT_DIM = 3
    stn3d = STNkd(k=INPUT_DIM)
    #print(stn3d)
    data = torch.rand(50, INPUT_DIM, 100) # num_sample, num_feature, num_point
    #data.size()
    input_trans = stn3d(data)
    fnet = PointNetFeat()
    feature = fnet(data)
    #out.shape
    ClsNet = PointNetCls(num_classes = 10)
    cls_out = ClsNet(data)
    print(cls_out.max(-1)[0].shape)
    SegNet = PointNetSeg(num_parts=40)
    seg_out = SegNet(data)
    print(seg_out.max(-1)[0].shape)