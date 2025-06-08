import torch
import torch.nn as nn
import torch.nn.parallel
import sys
import torch.nn.functional as F
sys.path.append("..")

def get_attention_map(gradients, activations):
    # 计算Grad-CAM的权重
    if gradients.dim() == 4:
        alpha = gradients.mean(dim=(2, 3), keepdim=True)  # 计算全局池化后的梯度
    else:
        alpha = gradients
    attention_map = F.relu((alpha * activations).sum(dim=1))  # 权重与特征图相乘后求和，激活函数ReLU
    attention_map = attention_map / (attention_map.max() + 1e-10)  # 归一化
    return attention_map


def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        #print('BatchNorm initial')
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


def weights_init_WD(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    #print("mod=", mod)
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        #mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data, gain=1)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    #ndf是输出的channel个数
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()# model模型

        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))# （32+2×1-4）/2+1=16 #wgan-gp kernel是3###第一个ndf是nc
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf# 图像的大小缩小两倍  channel数量不变 16对应64

        # Extra layers
        for t in range(n_extra_layers):#没有额外的卷积层
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4: # 图像大于4的话就继续 16 8 4 一共新加两层卷积层
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2# channel 变为2倍
            csize = csize / 2 # 图像缩小两倍

        # state size. K x 4 x 4 #最后一层卷积  一共四层卷积
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))# 图像大小现在已经小于4了 (（3）+2×0-4）/2+1=1  nz=100

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4# ngf=64  图像大小      32个channel对应4的图像大小
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2 # 配合前面

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())#逐元素
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



def sampling(z_mean_tmp, z_log_var_tmp, opt_tmp):
    z_mean, z_log_var, opt = z_mean_tmp, z_log_var_tmp, opt_tmp
    device = torch.device("cuda:{}".format(opt_tmp.gpu_ids[0]) if opt_tmp.device != 'cpu' else "cpu")
    #print("z_mean=",z_mean.view(-1,opt.nz))
    epsilon=torch.randn(size=(z_mean.view(-1,opt.nz).shape[0], opt.nz)).to(device)
    #epsilon = K.random_normal(shape=(K.shape(z_mean)[0], opt.nz))  # 采样  返回形状 tensor z 空间采样  batchsize  w×h×c   两个latentdim
    return z_mean + torch.exp(z_log_var / 2) * epsilon  ## 输出的是logsigma平方

class Gaussian(nn.Module):
    def __init__(self, opt):
        super(Gaussian, self).__init__()
        self.num_classes = opt.num_classes
        self.mean = torch.nn.Parameter(torch.zeros(size=(self.num_classes, opt.nz), requires_grad=True))#不用在定义成为variable
    def forward(self, z):
        z = torch.unsqueeze(z,1)
        mean1 = torch.unsqueeze(self.mean,0)
        return z - mean1

class y_classfier(nn.Module):
    def __init__(self, opt):
        super(y_classfier, self).__init__()

        self.linear1 = nn.Linear(opt.nz, opt.nz)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(opt.nz, int(opt.nz/2.0))
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.linear3 = nn.Linear(int(opt.nz/2.0), opt.num_classes)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.Softmax = nn.Softmax(dim=1)

        self.opt = opt

    def forward(self, input):

        y1=self.linear1(input.view(-1,self.opt.nz))
        y1=self.leaky_relu1(y1)
        y2=self.linear2(y1.view(-1,self.opt.nz))
        y2=self.leaky_relu2(y2)
        y3=self.linear3(y2.view(-1,int(self.opt.nz/2.0)))
        y3=self.leaky_relu3(y3)
        h = self.Softmax(y3)
        return h

class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.z_mean = nn.Linear(opt.nz, opt.nz)  # 每个独立z的均值
        self.z_log_var = nn.Linear(opt.nz, opt.nz)  # 每个独立z的方差
        self.opt = opt

    def forward(self, x):
        # 编码器前向传播
        encoded_features = self.encoder(x)
        z_mean, z_log_var = self.get_latent(encoded_features)

        # 采样 z，并确保其梯度计算
        z_added_noise = sampling(z_mean, z_log_var, self.opt)
        z_added_noise.requires_grad_(True)

        # 解码器前向传播
        gen_image = self.decoder(z_added_noise.view(-1, self.opt.nz, 1, 1))
        return gen_image, z_added_noise, z_mean, z_log_var

    def get_latent(self, latent_i):
        z_mean = self.z_mean(latent_i.view(-1, self.opt.nz))
        z_log_var = self.z_log_var(latent_i.view(-1, self.opt.nz))
        return z_mean, z_log_var

    def compute_attention_map(self, loss, z_added_noise):
        # 计算 z_added_noise 的梯度
        gradients = torch.autograd.grad(outputs=loss, inputs=z_added_noise, retain_graph=True, create_graph=True)[0]
        # 使用梯度和激活值生成注意力图
        attention_map = get_attention_map(gradients, z_added_noise)
        return attention_map


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


class Encoder_without_batchnorm(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    #ndf是输出的channel个数
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder_without_batchnorm, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf), nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf), nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf), nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat), nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))# 图像大小现在已经小于4了 (（3）+2×0-4）/2+1=1  nz=100

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Net_W_D(nn.Module):

    def __init__(self, opt):
        super(Net_W_D, self).__init__()
        self.model = Encoder_without_batchnorm(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.layers = list(self.model.main.children())
        self.features = nn.Sequential(*self.layers[:-1])
        self.classifier = nn.Sequential(self.layers[-1], nn.Flatten())  # nz = 1 (0): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(opt.nz, opt.nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nz, opt.nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.nz, 1)
        )
        self.nz = opt.nz

    def forward(self, x):
        latent_i = self.features(x)
        feature = self.classifier(latent_i)
        w_dis = self.output_layer(feature)

        return w_dis, feature


class D_net_gauss(nn.Module):
    def __init__(self,opt):
        super(D_net_gauss, self).__init__()
        self.N = 1000
        self.lin1 = nn.Linear(opt.nz, self.N)
        self.lin2 = nn.Linear(self.N, self.N)
        self.lin3 = nn.Linear(self.N, 1)
        self.opt = opt

    def forward(self, x):
        x = F.dropout(self.lin1(x.view(-1, self.opt.nz)), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x.view(-1, self.N)), p=0.2, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x.view(-1, self.N)))