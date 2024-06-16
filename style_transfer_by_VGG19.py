import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt, image as mpimg
from torch import nn
from d2l import torch as d2l
import torchvision.models as models


content_img = Image.open('rainier.jpg')    # Image.open()读出来的是pillow
# plt.imshow( content_img )
# plt.show()


style_img = Image.open('autumn-oak.jpg')  # Image.open()读出来的是pillow
# plt.imshow( np.array(style_img) )  # imshow()需要的是numpy？可是上面的plt.imshow( content_img )是可以的
# plt.show()
# #
# #
rgb_mean = torch.tensor([0.485, 0.456, 0.406])  # 炼丹
rgb_std = torch.tensor([0.229, 0.224, 0.225])   # 炼丹

def preprocess(img, image_shape):
    """
    对输入图像在RGB三个通道分别做标准化，
    并将结果变换成卷积神经网络接受的输入格式
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    return transforms(img).unsqueeze(0)


def post_process(img):
    """
    将输出图像中的像素值还原回标准化之前的值
    """
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# ################################################################################

# 假设你已经有了本地的VGG19权重文件路径
local_weights_path = r'vgg19-dcbb9e9d.pth'

# 加载没有预训练权重的VGG19模型结构
pretrained_net = models.vgg19(weights=None)

# 加载本地的权重文件
state_dict = torch.load(local_weights_path)
pretrained_net.load_state_dict(state_dict)
# print(pretrained_net)


style_layers = [3,8,15,22]
content_layers = [15]
net = nn.Sequential( *[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)] )

# print(net)

# # # 抽取 特征###########################################################################################

def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    """对内容图像抽取内容特征"""
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    """对风格图像抽取风格特征"""
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y



# loss ##############################################################################

def content_loss(Y_hat, Y):
    """内容损失"""
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    """风格损失"""
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


def tv_loss(Y_hat):
    """全变分损失"""
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


content_weight, style_weight, tv_weight = 1, 1e3, 10  # 风格转移的损失函数是内容损失、风格损失和总变化损失的加权和

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]

    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]

    tv_l = tv_loss(X) * tv_weight

    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])

    return contents_l, styles_l, tv_l, l

# #######################################################################################

class SynthesizedImage(nn.Module):
    """
    合成的图像是训练期间唯一需要更新的变量
    定义一个简单的模型SynthesizedImage，并将合成的图像视为模型参数。
    模型 前向传播只需返回模型参数即可
    """
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X, device, lr, styles_Y):

    gen_img = SynthesizedImage(X.shape).to(device)

    gen_img.weight.data.copy_(X.data)

    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr) # optim

    styles_Y_gram = [gram(Y) for Y in styles_Y]

    return gen_img(), styles_Y_gram, trainer


c_l=[]
s_l=[]
t_l=[]
loss=[]

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)

    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)

        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        ############################################
        # # contents_l、styles_l 都是 <class 'list'>
        c_l.append([tensor.item() for tensor in contents_l])
        s_l.append([tensor.item() for tensor in styles_l])
        # # # tv_l、l 都是<class 'torch.Tensor'>
        t_l.append(tv_l)
        loss.append(l)
        # ##########################################
        trainer.step()
        scheduler.step()

    return X


# TRAIN
device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)


pic = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
# type(pic) 是 <class 'torch.nn.parameter.Parameter'> (1, 3, 300, 450)，
# 表示一个样本，三个通道，以及 300x450 的高度和宽度


# t_l = [item.item() for item in t_l]
# loss = [item.item() for item in loss]
#
#
# fig = plt.figure()
# plt.plot(list(range(1, len(c_l) + 1)), np.array(c_l), color='red', marker='o', label='content_loss')
# plt.plot(list(range(1, len(s_l) + 1)), np.array(s_l), color='blue', marker='o', label='style_loss')
# plt.plot(list(range(1, len(t_l) + 1)), np.array(t_l), color='yellow', marker='o', label='tv_loss')
# plt.plot(list(range(1, len(loss) + 1)), np.array(loss), color='black', marker='o', label='loss')
# plt.legend()
# plt.savefig(r"loss.png")
# # plt.show()
# plt.close()



### show pic
fig = plt.figure()
plt.imshow(pic[0].cpu().detach().numpy().transpose(1, 2, 0) )  # pic张量从GPU移动到CPU,然后转换为NumPy数组,输入imshow()
##plt.imshow() 函数期望的输入形状应该是 (height, width) 或者 (height, width, channels)，而不是 (batch_size, channels, height, width)。
##您的张量 pic 的形状是 (1, 3, 300, 450)，pic[0]是取第一个样本，其通道数为 3，高度为 300，宽度为 450。
##为了正确地显示图像，您需要将其转换为适当的形状。然后将其转换为 (height, width, channels) 的形状
plt.savefig(r"out3.png")
plt.show()
plt.close()


