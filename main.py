
import torch
from PIL import Image
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
skew_320 = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    normalize,
])

model = torch.jit.load("sscd_disc_mixup.torchscript.pt")
img = Image.open("image_mmstar.jpg").convert('RGB')

batch1 = small_288(img).unsqueeze(0)
batch2 = skew_320(img).unsqueeze(0)

embedding1 = model(batch1)[0, :]
embedding2 = model(batch2)[0, :]

print(embedding1.shape, embedding2.shape)
print("Similarity:", embedding1.dot(embedding2).item())
print("Distance:", (embedding1-embedding2).norm().item())