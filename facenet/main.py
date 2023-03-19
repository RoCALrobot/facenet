import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

root_dir = '/home/ymliee/data/facenet/'
data_dir = root_dir+'train/'
model_dir = root_dir
batch_size = 32
epochs = 8

"""#### Determine if an nvidia GPU is available"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

"""#### Define MTCNN module

See `help(MTCNN)` for more details.
"""

from facenet.mtcnn import PNet, RNet, ONet,MTCNN
path_p_trained = model_dir+'pnet.pt'
path_r_trained = model_dir+'rnet.pt'
path_o_trained = model_dir+'onet.pt'
pnet = PNet(path_p_trained)
rnet = RNet(path_r_trained)
onet = ONet(path_o_trained)
mtcnn = MTCNN(
    pnet, rnet, onet,
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

"""#### Perfom MTCNN facial detection

Iterate through the DataLoader object and obtain cropped faces.
"""

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(root_dir, '/train_cropped'))
        for p, _ in dataset.samples
]

def collate_pil(x): 
    out_x, out_y = [], [] 
    for xx, yy in x: 
        out_x.append(xx) 
        out_y.append(yy) 
    return out_x, out_y 

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_pil
)

for i, (x, y) in enumerate(loader):
    print(x)
    print(y)
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
# Remove mtcnn to reduce GPU memory usage
del mtcnn

"""#### Define Inception Resnet V1 module

See `help(InceptionResnetV1)` for more details.
"""

from facenet.inception_resnet_v1 import InceptionResnetV1
from facenet.mtcnn import fixed_image_standardization
resnet = InceptionResnetV1(
    classify=True,
    pretrained='data/vggface2.pt',
    num_classes=len(dataset.class_to_idx)
).to(device)

"""#### Define optimizer, scheduler, dataset, and dataloader"""

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder('data/cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

"""#### Define loss and evaluation functions

#### Train model
"""

def train(model, loss_fn, loader, optimizer=None):
    mode = 'Train' if model.training else 'Valid'
    loss = 0
    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_batch = loss_fn(y_pred, y)

        if model.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
     
        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
  
    loss = loss / (i_batch + 1)
    print(loss.item())        
    return model, loss

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    train(
        resnet, loss_fn, train_loader, optimizer
    )

import pandas as pd
import matplotlib.pyplot as plt
aligned = []
names = []
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('data/img')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn)

for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print(x_aligned.shape)
        print('Face detected with probability: {:8f}'.format(prob.max()))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

print(names)
resnet.eval()
for x in aligned:
    x = x.to(device)
    plt.imshow(x.permute(1,2,0))
    plt.show()
    pred = resnet(x.unsqueeze(0))
    print(names[pred[0].argmax(0)])
