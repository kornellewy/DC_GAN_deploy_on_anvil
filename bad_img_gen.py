from __future__ import print_function
import time
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from pathlib import Path
# kjn
from generator_model import Generator
from discriminator_model import Discriminator
#anvil
import anvil.server
import anvil.media




# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

model = Generator(0)
model.load_state_dict(torch.load('model/netG.pth'))
model.eval()
print("wczytano model")

def gen_img():
	print("generacja zdjecia")
	noise = torch.randn(1, nz, 1, 1, device=device)
	fake = model(noise)
	plt.figure(figsize=(64,64))
	plt.axis("off")
	plt.title("Generated Image")
	plt.imshow(transforms.ToPILImage()(torch.mean(fake, 0)))
	plt.savefig('img/fake.png')

@anvil.server.callable
def load_img():
	start_time = time.time()
	gen_img()
	media_object = anvil.media.from_file('img/fake.png')
	elapsed_time = time.time() - start_time
	print(elapsed_time)
	print("wysylanie img")
	return media_object


anvil.server.connect("")
anvil.server.wait_forever()
