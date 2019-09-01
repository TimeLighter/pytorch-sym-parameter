#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset
import utils_pl
from vgg import Vgg16
import networks
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/photo2ukiyoe2vangogh/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--log_int', type=int, default=50, help='number of cpu threads to use during batch generation')
parser.add_argument('--style_image', type=str, default='images/style-images/rain-princess.jpg', help='root directory of the dataset')
parser.add_argument("--rc_weight", type=float, default=2.,
                                  help="reconstruction weight, default is 2")
parser.add_argument("--identity_weight", type=float, default=5.,
                                  help="weight for identity-loss, default is 5.")

parser.add_argument("--content_weight", type=float, default=1e-3,
                                  help="weight for content-loss, default is 1e-3")
parser.add_argument("--style_weight", type=float, default=1e2,
                                  help="weight for style-loss, default is 1e2")
parser.add_argument("--style_size", type=int, default=256,
                                  help="size of style-image, default is the original size of style image")

parser.add_argument("--alpha", type=float, default=0.5,
                                  help="alpha for dirichlet")
parser.add_argument('--output_dir', type=str, default='output_final', help='output directory')

parser.add_argument('--data_order', nargs='+', default=['A','B', 'C'])

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG = Generator(opt.output_nc, opt.input_nc)
# netG = GeneratorNORM(opt.output_nc, opt.input_nc)
# netG = GeneratorCat(opt.output_nc, opt.input_nc)
netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, False)

device = 'cpu'
if opt.cuda:
    netG.cuda()
    netD_B.cuda()
    device = 'cuda'

# netG.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess

criterionGAN = networks.GANLoss().to(device)
criterion_MSE = torch.nn.MSELoss(reduce=False)
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

############### only use B to A ###########################
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensorreal_A

input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
fake_C_buffer = ReplayBuffer()
# Dataset loader
transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True, order=opt.data_order),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), opt.log_int)
###################################

beta = torch.distributions.beta.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([opt.alpha, opt.alpha, opt.alpha]))

ones = Variable(torch.ones(1,1).cuda())
A1B0C0 = torch.cat([ones * 1.0, ones * 0.0, ones * 0.0], 1)
A0B1C0 = torch.cat([ones * 0.0, ones * 1.0, ones * 0.0], 1)
A0B0C1 = torch.cat([ones * 0.0, ones * 0.0, ones * 1.0], 1)
A05B05C0 = torch.cat([ones * 0.5, ones * 0.5, ones * 0.0], 1)
A0B05C05 = torch.cat([ones * 0.0, ones * 0.5, ones * 0.5], 1)
A05B0C05 = torch.cat([ones * 0.5, ones * 0.0, ones * 0.5], 1)

scale = 1.0
STYLE_WEIGHTS = np.array([1e-1, 1, 1e1, 5]) * opt.style_weight

### for perceptual loss ###

vgg = Vgg16(requires_grad=False).to(device)
style_transform = transforms.Compose([
    transforms.ToTensor(),
])
style = utils_pl.load_image(opt.style_image, size=opt.style_size)
style = style_transform(style)
style = style.repeat(opt.batchSize, 1, 1, 1).to(device)

features_style = vgg(utils_pl.normalize_batch(style))
gram_style = [utils_pl.gram_matrix(y) for y in features_style]

### for perceptual loss ###

for i, batch in enumerate(dataloader):
    real_C = Variable(input_C.copy_(batch['C']))
    real_perceptual = vgg(real_C.detach())
    break
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        real_C = Variable(input_C.copy_(batch['C']))
        batch_size = real_A.size()[0]
        cond = Variable(dirichlet.sample_n(batch_size)).cuda()
        cond0 = cond[:, 0]
        cond1 = cond[:, 1]
        cond2 = cond[:, 2]
        #################### Generator ####################
        optimizer_G.zero_grad()
        out_im = netG(real_A, cond)

        ########## loss A Reconstruction ##########
        loss_RC = torch.abs(out_im.view(batch_size, -1) - real_A.view(batch_size, -1) )
        # loss_RC = (fake_A.view(batch_size, -1) - real_B.view(batch_size, -1))**2

        loss_RC = loss_RC.mean(1) * opt.rc_weight
        ########## loss A Reconstruction ##########


        ########## loss B GAN ##########
        pred_fake = netD_B(out_im)
        # loss_GAN = criterion_MSE(pred_fake.squeeze(), target_real).view(batch_size, -1).mean(1)
        loss_GAN = criterionGAN(pred_fake, True)

        # # Identity loss
        same_B = netG(real_B, cond)
        loss_identity = torch.abs(same_B.view(batch_size, -1) - real_B.view(batch_size, -1)).mean(1)

        loss_GANB = loss_GAN + opt.identity_weight*loss_identity
        ########## loss B GAN ##########


        ########## loss C Perceptual ##########
        pF = utils_pl.normalize_batch((out_im+1.0) * 0.5)
        pR = utils_pl.normalize_batch((real_A+1.0) * 0.5)

        features_F = vgg(pF)
        features_R = vgg(pR)

        content_loss = opt.content_weight * criterion_MSE(features_F.relu2_2, features_R.relu2_2).view(batch_size, -1).mean(1)

        style_loss = 0.
        for l, weight in enumerate(STYLE_WEIGHTS):
            gram_s = gram_style[l]
            gram_y = utils_pl.gram_matrix(features_F[l])
            style_loss += float(weight) * criterion_MSE(gram_y, gram_s.expand_as(gram_y)).view(batch_size, -1).mean(1)

        # for ft_y, gm_s in zip(features_F, gram_style):
        #     gm_y = utils_pl.gram_matrix(ft_y)
        #     target = gm_s[:batch_size, :, :]
        #     style_loss += criterion_MSE(gm_y, target).view(batch_size, -1).mean(1)
        # style_loss *= opt.style_weight

        diff_i = torch.sum(torch.abs(out_im[:, :, :, 1:] - out_im[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(out_im[:, :, 1:, :] - out_im[:, :, :-1, :]))
        tv_loss = (opt.style_weight*1e-10) * (diff_i + diff_j)

        loss_P = content_loss + style_loss # + tv_loss
        ########## loss C Perceptual ##########


        ########## total loss ##########
        loss_G = torch.mean(cond0*loss_RC + cond1*loss_GANB + cond2*loss_P )
        # loss_G = loss_P
        loss_G.backward()
        optimizer_G.step()


        ##########  Discriminator B ##########
        optimizer_D_B.zero_grad()

        # Fake loss
        fake_B, cond_r = fake_A_buffer.push_and_pop((out_im, cond))
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # loss_D_fake = criterion_MSE(pred_fake.squeeze(), target_fake)

        # Real loss
        pred_real = netD_B(real_B)
        # loss_D_real = criterion_MSE(pred_real.squeeze(), target_real)
        loss_D_real = criterionGAN(pred_real, True)

        # Total loss
        loss_D_A = torch.mean(cond_r[:, 1] * (loss_D_real + loss_D_fake) ) * 0.5
        loss_D_A.backward()

        optimizer_D_B.step()

        # Progress report (http://localhost:8097)
        if (i+1)%opt.log_int==0:
            # print('loss_G: {}'.format(loss_G), 'loss_G_RC: {}'.format(loss_RC),
            #             'loss_G_GAN: {}'.format(loss_GAN),'loss_D: {}'.format(loss_D_A),
            #             'cond: {}'.format(cond[0].data.cpu()) )
            with torch.no_grad():
                netG.eval()
                fake_A1B0C0 = netG(real_A, A1B0C0.repeat(batch_size, 1))
                fake_A0B1C0 = netG(real_A, A0B1C0.repeat(batch_size, 1))
                fake_A0B0C1 = netG(real_A, A0B0C1.repeat(batch_size, 1))
                fake_A05B05C0 = netG(real_A, A05B05C0.repeat(batch_size, 1))
                fake_A0B05C05 = netG(real_A, A0B05C05.repeat(batch_size, 1))
                fake_A05B0C05 = netG(real_A, A05B0C05.repeat(batch_size, 1))

                print("cond %.3f"%(cond1[0].data.cpu()))

                logger.log({'loss_G': loss_G, 'loss_G_RC': (loss_RC),
                            'loss_G_GAN': (loss_GANB),
                            'loss_perceptual': (loss_P),
                            'loss_D': (loss_D_A),
                            'cond': (cond1[0].squeeze())
                            },

                           images={'real_A': real_A,
                                   'A1B0C0': fake_A1B0C0, 'A0B1C0': fake_A0B1C0, 'A0B0C1': fake_A0B0C1,
                                   'A05B05C0': fake_A05B05C0, 'A0B05C05': fake_A0B05C05, 'A05B0C05': fake_A05B0C05,
                                   'real_B': real_B
                                   }, )

                netG.train()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG.state_dict(), opt.output_dir+'/netG_%02d.pth'%(epoch))
    torch.save(netD_B.state_dict(), opt.output_dir+'/netD_B.pth')
###################################
