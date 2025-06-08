from collections import OrderedDict
import os
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from networks import NetG, weights_init, Net_W_D, weights_init_WD, sampling, Gaussian, y_classfier, latent_loss, get_attention_map
from loss import l2_loss
from torch.optim import lr_scheduler

##
class  AG_CVAEGAN(object):

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'AG_CVAEGAN'

    def __init__(self, opt, train_dataloader=None, param=None):
        super(AG_CVAEGAN, self).__init__()
        ##
        # Initalize variables.
        self.opt = opt
        self.param = param
        self.train_dataloader = train_dataloader

        self.D_count=0
        self.CRITIC_ITERS = self.opt.CRITIC_ITERS
        self.CRITIC_ITERS2 = self.opt.CRITIC_ITERS2
        self.CRITIC_ITERS3 = self.opt.CRITIC_ITERS3
        #self.device = torch.device("cuda:2" if self.opt.device != 'cpu' else "cpu")
        self.device = torch.device("cuda:{}".format(self.opt.gpu_ids[0]) if self.opt.device != 'cpu' else "cpu")

        self.fake = None
        self.latent_i = None#z_mean
        self.latent_o = None#z_log_var
        self.D_w_cost = None
        self.Wasserstein_D = None
        self.G_w_cost = None
        self.kl_loss = None
        self.cat_loss = None

        # -- Generator attributes.
        self.err_g_l1l = None


        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.best_auc = 0

        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.net_w_d = Net_W_D(self.opt).to(self.device)
        self.net_Gaussian = Gaussian(self.opt).to(self.device)
        self.net_y_classfier = y_classfier(self.opt).to(self.device)


        # 高斯那个已经初始化了
        self.net_y_classfier.apply(weights_init)
        self.netg.apply(weights_init)
        self.net_w_d.apply(weights_init_WD)


        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.net_w_d.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")


        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.l2l_criterion = l2_loss

        ##
        # Initialize input tensors.#创建初始的输入
        self.input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        ##mnist 8 8 10 cifar 2 2 4
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.net_w_d.train()
            self.net_Gaussian.train()
            self.net_y_classfier.train()
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr / 2, betas=(self.opt.beta1, 0.999))

            self.optimizer_w_d = optim.Adam(self.net_w_d.parameters(), lr=self.opt.lr / 1,
                                            betas=(self.opt.beta1, 0.999))
            self.optimizer_w_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr / 1, betas=(self.opt.beta1, 0.999))


            self.optimizer_y_classfier = optim.Adam([{'params': self.net_Gaussian.parameters()},
                                                     {'params': self.net_y_classfier.parameters()},
                                                  {'params': self.netg.encoder.parameters()},
                                                  {'params': self.netg.z_mean.parameters()},
                                                  {'params': self.netg.z_log_var.parameters()}], lr=self.opt.lr / 4,
                                                 betas=(self.opt.beta1, 0.999))

            self.optimizer_enc = optim.Adam(self.netg.encoder.parameters(), lr=self.opt.lr / 2,
                                            betas=(self.opt.beta1, 0.999))

            def lambda_rule(epoch):#epoach自带的变量
                lr_l = 1.0 - max(0, epoch + 1  - self.opt.iter - 10 ) / float(self.opt.niter_decay)#从30开始以1/200斜率线性缩减
                return lr_l

            self.scheduler_optimizer_g = lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=lambda_rule)
            self.scheduler_optimizer_w_d = lr_scheduler.LambdaLR(self.optimizer_w_d, lr_lambda=lambda_rule)
            self.scheduler_optimizer_w_g = lr_scheduler.LambdaLR(self.optimizer_w_g, lr_lambda=lambda_rule)
            self.scheduler_optimizer_enc = lr_scheduler.LambdaLR(self.optimizer_enc, lr_lambda=lambda_rule)
            self.scheduler_optimizer_y_classfier = lr_scheduler.LambdaLR(self.optimizer_y_classfier, lr_lambda=lambda_rule)#不用
    def set_input(self, input):

        with torch.no_grad():
            # 调整 self.input 和 self.gt 的大小并复制 input[0] 和 input[1]
            self.input.resize_(input.size()).copy_(input)  # 0 表示数据


    def update_w_netd(self):
        self.net_w_d.zero_grad()
        out_d_real, _ = self.net_w_d(self.input)
        out_d_real = out_d_real.mean()

        self.fake, self.z_added_noise, self.latent_i, self.latent_o = self.netg(self.input)

        out_d_fake, _ = self.net_w_d(self.fake.detach())
        out_d_fake = out_d_fake.mean()

        gradient_penalty = self.calc_gradient_penalty(self.net_w_d, self.input, self.fake)
        self.D_w_cost = out_d_fake - out_d_real + gradient_penalty
        self.D_w_cost.backward()
        self.Wasserstein_D = out_d_real - out_d_fake
        self.optimizer_w_d.step()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print "real_data: ", real_data.size(), fake_data.size()
        BATCH_SIZE = self.opt.batchsize
        alpha = torch.ones(BATCH_SIZE, 1)  ###收敛太快，太像和不像都不好 因此取0.5  1最快
        alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, self.opt.nc, self.opt.isize, self.opt.isize)
        alpha = alpha.cuda(self.device) if 1 else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if 1:
            interpolates = interpolates.cuda(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates,_ = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                      self.device) if 1 else torch.ones( disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty


    def update_w_netg_auto(self):
        """
        Update Generator with integrated Attention Expansion Loss.
        """
        self.netg.zero_grad()
        # 前向传播并获取必要的变量
        self.fake, _, self.latent_i, self.latent_o = self.netg(self.input)

        # 计算基础损失
        self.err_g_l1l = self.l1l_criterion(self.fake, self.input)  # constrain x' to look like x
        self.err_g_l1l.backward()
        self.optimizer_g.step()

    def update_attention_expansion_loss(self):
        """
        更新编码器的参数，优化Attention Expansion Loss
        """
        # 清零编码器优化器的梯度
        self.netg.zero_grad()

        # 前向传播获取所需的变量
        self.fake, self.z_added_noise, self.latent_i, self.latent_o = self.netg(self.input)
        self.err_g_l1l = self.l1l_criterion(self.fake, self.input)
        # 计算注意力扩展损失
        attention_map = self.netg.compute_attention_map(self.err_g_l1l, self.z_added_noise)
        attention_expansion_loss = (1 - attention_map).mean()
        attention_expansion_loss.backward()

        # 更新编码器参数
        self.optimizer_enc.step()

        # 保存损失值
        self.attention_loss = attention_expansion_loss


    def update_y_classifier(self):
        self.netg.encoder.zero_grad()
        self.netg.z_mean.zero_grad()
        self.netg.z_log_var.zero_grad()
        self.net_y_classfier.zero_grad()
        self.net_Gaussian.zero_grad()

        _, z_added_noise, self.latent_i, self.latent_o = self.netg(self.input)
        z_prior_mean = self.net_Gaussian(z_added_noise)
        y = self.net_y_classfier(z_added_noise)
        # 计算 KL 散度损失
        M = torch.zeros(1, self.opt.nz)
        kl_loss = - 0.5 * (self.latent_o.unsqueeze(1) - torch.pow(z_prior_mean,2))
        self.kl_loss = torch.mean(torch.addbmm(M, y.unsqueeze(1).cpu(), kl_loss.cpu())/self.opt.batchsize).to(self.device)

        # 计算分类损失
        self.cat_loss = torch.mean(torch.mean(y * torch.log(y + 1e-15),0))

        total_latent_loss = 1 * self.cat_loss + 1 * self.kl_loss
        total_latent_loss.backward()
        self.optimizer_y_classfier.step()

    def update_w_netg(self):
        self.netg.zero_grad()

        #####原始
        self.fake, _, _,_ = self.netg(self.input)
        out_g, _ = self.net_w_d(self.fake)
        out_g = -torch.squeeze(out_g).mean()
        out_g.backward(retain_graph=True)
        self.optimizer_w_g.step()
        self.G_w_cost = out_g

    ##
    def optimize(self):
        """ Optimize netD and netG  networks.
        """
        if self.D_count<self.CRITIC_ITERS:#一起训练5步3
            self.update_w_netd()

            self.update_w_netg_auto()
            self.update_attention_expansion_loss()
            self.update_y_classifier()

        elif self.D_count<self.CRITIC_ITERS2: #gan_d训练4步3
            self.update_w_netd()
        elif self.D_count<self.CRITIC_ITERS3: #gan_g训练1步1
            self.update_w_netg()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('discriminator_cost', self.D_w_cost.item()),
            ('generator_cost', self.G_w_cost.item()),
            ('Wasserstein_distance', self.Wasserstein_D.item())]
        )

        errors2 = OrderedDict([
            ('cat_loss',self.cat_loss.item()),
            ('l1_loss', self.err_g_l1l.item())]
        )
        return errors, errors2

    def train_epoch(self):
        """
        Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        self.D_count = 0

        for i, data in enumerate(tqdm(self.train_dataloader, leave=False, total=len(self.train_dataloader),desc='Training')):
            self.total_steps += self.opt.batchsize # 取多少个样本
            epoch_iter += self.opt.batchsize # 取多少个样本
            self.set_input(data)
            self.optimize()
            if self.D_count == self.CRITIC_ITERS3 - 1:
                self.D_count = 0
            else:
                self.D_count += 1
            if self.total_steps % self.opt.print_freq == 0:
                self.get_errors()

        print(">> Training model %s. Epoch %d/%d" % (self.name(), self.epoch+1, self.opt.niter))

    def update_lr(self):
        self.scheduler_optimizer_w_d.step()#更新wgand的权重衰减
        self.scheduler_optimizer_w_g.step()#更新wgang的权重衰减
        self.scheduler_optimizer_g.step()#更新自编码器的权重衰减
        self.scheduler_optimizer_y_classfier.step()#暂时不用

        print("optimizer_w_d.param_groups=", self.optimizer_w_d.param_groups[0]['lr'])
        print("optimizer_w_g.param_groups=", self.optimizer_w_g.param_groups[0]['lr'])
        print("optimizer_g.param_groups=", self.optimizer_g.param_groups[0]['lr'])
        print("optimizer_y_classfier.param_groups=", self.optimizer_y_classfier.param_groups[0]['lr'])

    def train(self):
        self.total_steps = 0
        self.best_auc = 0
        self.best_auc_pr = 0
        self.best_f1_score = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name())
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_epoch()
            self.update_lr()
        print(">> Training model %s.[Done]" % self.name())



