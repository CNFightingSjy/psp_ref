from models.stylegan2.model import Discriminator
import torch
import numpy as np

def load_model(path):

    models = torch.load(path)
# print(models['g'])
# a = torch.randn(10,10).cuda()
# print(torch.cuda.is_available())
# print(d.keys())
    print(models.keys())
    # for k, v in models:
    #     print(k)

load_model("/data/shijianyang/code/pixel2style2pixel/pretrained_models/stylegan2-ffhq-config-f.pt")
load_model("/data/shijianyang/code/pixel2style2pixel/pretrained_models/060000.pt")
# d = Discriminator(256, 2).to("cuda:1")
# print(d)
# for n, p in d.named_parameters():
#     print(n,":", p.type())

# a = np.random.randn(1, 224, 224)
# a = np.random.randn(128 ,3 ,1 ,1)
# b = torch.from_numpy(a)
# b = b.to("cuda:1").float()
# b = torch.squeeze(b)
# conv = torch.nn.Conv2d(1, 3, 3 ,1 ,1).to("cuda:1")
# c = conv(b)
# print(b.shape)
# b = b.unsqueeze(0)
# b = torch.DoubleTensor(b)
# print(b)

# out = d(b.to("cuda:1").float())
# print(out)