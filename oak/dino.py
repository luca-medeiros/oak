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


def upload_collection(qdrant_client, collection_name, features, payload):
    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=features,
        payload=payload,
        ids=None,
        batch_size=256
    )


def search(qdrant_client, collection_name, features, top_k=2):
    from qdrant_client.http import models
    search_queries = [models.SearchRequest(
            vector=k,
            with_payload=True,
            limit=top_k
        ) for k in  features.tolist()]
    search_result = qdrant_client.search_batch(
        collection_name=collection_name,
        requests=search_queries,
        with_payload=True
        
    )
    return search_result

if __name__ == '__main__':
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    
    qdrant_client = QdrantClient(
        url="https://40eba53d-2c28-4027-b1b9-04c74cab045b.us-east-1-0.aws.cloud.qdrant.io:6333",
        api_key="K6KGkfEUqNOpBT5xGloGOrjHdMoAnbnrPM16YFyiE39Bhqsx6NPDPA",
    )

    qdrant_client.recreate_collection(
        collection_name="oak",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    image_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
    dataset = torchvision.datasets.ImageFolder(root='/home/nuvilab/internal_services/oak/tiny-coco/small_coco', transform=image_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dino = DINOv2()
    features = dino.extract_features(dataloader)
    payloads = [{'image_path': image_path} for image_path, _ in dataset.samples]
    upload_collection(qdrant_client, 'oak', features, payloads)
    a = 1
    # D, I = knn(features, 4)
    # a = 1
