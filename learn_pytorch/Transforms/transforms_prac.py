from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter('logs')
img_path = 'data/train/ants_image/0013035.jpg'
img = Image.open(img_path)

# 将图片转化成tensor形式
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
writer.add_image('ToTensor', img_tensor, 1)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('normalize', img_norm, 1)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> Resize -> img_resize PIL
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
writer.add_image('resize', img_resize, 1)

writer.close()