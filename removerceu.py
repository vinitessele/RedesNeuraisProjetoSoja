import torch
import torchvision
from PIL import Image
import numpy as np

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

img = Image.open(r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\imagens\uploads\IMG-20251112-WA0053__V3_V4.jpeg").convert("RGB")
t = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
x = t(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)["out"][0]

# classe 21 = céu (na COCO)
sky_mask = out.argmax(0).byte().cpu().numpy() == 21

img_np = np.array(img)
img_np[sky_mask] = 0  # zera o céu

Image.fromarray(img_np).save("sem_ceu.png")
