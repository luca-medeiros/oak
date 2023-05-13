import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import hubconf
import faiss


class DINOv2(nn.Module):

    def __init__(self, vit_arch: str = 'vit_large', device: str = 'cuda:0'):
        super().__init__()
        self.vit_arch = vit_arch
        self.device = device
        self.build_model()

    def build_model(self):
        self.model = hubconf._make_dinov2_model(arch_name=self.vit_arch)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract_features(self, dataloader):
        features = []

        with torch.no_grad():
            for images, class_index in dataloader:
                images = images.to(self.device)
                features.append(self.forward(images).detach().cpu())

        features = torch.cat(features)

        return features.numpy()
    
    def forward(self, x):
        return self.model(x)

def knn(features, k):
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    D, I = index.search(features, k)
    return D, I


if __name__ == '__main__':
    import cv2
    image_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
    dataset = torchvision.datasets.ImageFolder(root='/project/studios/this_studio/oak/small_coco/datalo', transform=image_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dino = DINOv2()
    features = dino.extract_features(dataloader)
    D, I = knn(features, 4)
    a = 1
