from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
img_path = 'data/train/bees_image/17209602_fe5a5a746f.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

# def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'): CWH就是channel, height, wide
writer.add_image('train', img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar('y=x', i, i)

writer.close()