#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from model import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/vangogh2photo', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator', type=str, default='pretrained/Model1.pth', help='generator checkpoint file')
parser.add_argument("--alpha", type=float, default=1.0,
                                  help="alpha for dirichlet")
parser.add_argument('--output_dir', type=str, default='output', help='output directory')
parser.add_argument('--data_order', nargs='+', default=['B', 'A'])

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

test_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.5],
        [0.33, 0.33, 0.33],
    ]
)

test_mat = Variable(test_mat).cuda()

ones = Variable(torch.ones(1,1).cuda())
A1B0C0 = torch.cat([ones * 1.0, ones * 0.0, ones * 0.0], 1)
A0B1C0 = torch.cat([ones * 0.0, ones * 1.0, ones * 0.0], 1)
A0B0C1 = torch.cat([ones * 0.0, ones * 0.0, ones * 1.0], 1)
A05B05C0 = torch.cat([ones * 0.5, ones * 0.5, ones * 0.0], 1)
A0B05C05 = torch.cat([ones * 0.0, ones * 0.5, ones * 0.5], 1)
A05B0C05 = torch.cat([ones * 0.5, ones * 0.0, ones * 0.5], 1)

with torch.no_grad():
    ###### Definition of variables ######
    # Networks
    net_G = Generator(opt.output_nc, opt.input_nc)
    # net_G = GeneratorCat(opt.output_nc, opt.input_nc)
    # net_G = GeneratorNORM(opt.output_nc, opt.input_nc)

    if opt.cuda:
        # netG_A2B.cuda()
        net_G.cuda()

    # Load state dicts
    net_G.load_state_dict(torch.load(opt.generator))

    # Set model's test mode
    net_G.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', order=opt.data_order),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    for i, batch in enumerate(dataloader):
        real_A = Variable(batch['A']).cuda()

        input_image = real_A.repeat(7, 1, 1, 1)
        image = 0.5*(net_G(input_image, test_mat).data + 1.0)
        save_image(image, opt.output_dir+'/%04d.png' % (i + 1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################
