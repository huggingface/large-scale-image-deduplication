import torch
from torchvision import transforms
from datasets import load_dataset

dataset = load_dataset("Lin-Chen/MMStar")
model = torch.jit.load("models/sscd_disc_mixup.torchscript.pt")

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

image = dataset['val'][0]['image'].convert('RGB')

small_image = small_288(image)
square_image = skew_320(image)

batch = torch.stack([small_image, small_image])

embeddings = model(batch)
print(embeddings.shape)
print("Similarity:", embeddings[0].dot(embeddings[1]).item())
print("Distance:", (embeddings[0]-embeddings[1]).norm().item())

batch1 = small_288(image).unsqueeze(0)
batch2 = skew_320(image).unsqueeze(0)

embedding1 = model(batch1)[0, :]
embedding2 = model(batch2)[0, :]

print(embedding1.shape, embedding2.shape)
print("Similarity:", embedding1.dot(embedding2).item())
print("Distance:", (embedding1-embedding2).norm().item())

