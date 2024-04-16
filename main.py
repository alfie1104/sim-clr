# https://jimmy-ai.tistory.com/312
from sklearn.datasets import fetch_openml
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import style


# GPU 사용 지정
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# [Step1] 데이터셋 불러오기
mnist = fetch_openml('mnist_784')

# 7만개 중 앞 6만개를 학습용 뒤 1만개를 테스트용으로 구성
# 이미지 픽셀수와 CNN의 input 형태를 고려하여 각 이미지를  1*28*28로 변환
X_train = torch.tensor(np.array(mnist.data)).float().reshape(-1,1,28,28)[:60000].to(device)
y_train = torch.tensor(np.array(list(map(np.int_, mnist.target))))[:60000].to(device)

X_test = torch.tensor(np.array(mnist.data)).float().reshape(-1,1,28,28)[60000:].to(device)
y_test = torch.tensor(np.array(list(map(np.int_, mnist.target))))[60000:].to(device)

print(X_train.shape) # torch.Size([60000, 1, 28, 28])
print(y_train.shape) # torch.Size([60000])

print(X_test.shape) # torch.Size([10000, 1, 28, 28])
print(y_test.shape) # torch.Size([10000])

# [Step2] Data Augmentation 구현
# 랜덤으로 10*10 픽셀 부위를 골라 회색으로 마킹한 뒤 왼쪽으로 90도 만큼 회전하는 Data Augmentation 수행
def cutout_and_rotate(image):
    image = image.clone().detach() # 얕은 복사 문제 주의(원본 유지) (detach는 현재 tensor graph에서 새로운 tensor를 반환함)
    x_start = np.random.randint(20) # cut out 시작할 x축 위치(0~19중 1개)
    y_start = np.random.randint(20) # cut out 시작할 y축 위치(0~19중 1개)

    image[..., x_start:x_start+9, y_start:y_start+9] = 255 /2 # x_start ~ x_start + 9, y_start ~ y_start+9 픽셀 범위를 회색으로 마킹
    return torch.rot90(image, 1, [-2, -1]) # 마지막 2개의 axis(뒤에서 2번째, 뒤에서 1번째 차원) 기준 90도 회전

# [Option] Data Augmentation 결과 확인
def show_augmentation_result():
    # 흰색 배경 및 크기 지정
    style.use('default')
    figure = plt.figure()
    figure.set_size_inches(4,2)

    # 흑백으로 출력하기 위한 스타일 설정
    style.use('grayscale')

    # 1*2 사이즈의 격자 설정
    axes = []
    for i in range(1,3):
        axes.append(figure.add_subplot(1,2,i))

    # 첫 이미지에 대한 원본 이미지 및 augmentation 수행된 이미지 시각화
    img_example = X_train[0].clone().detach().cpu()
    original = np.array(img_example).reshape(-1, 28).astype(int)
    aug_img = np.array(cutout_and_rotate(img_example)).reshape(-1,28).astype(int)

    axes[0].matshow(original)
    axes[1].matshow(aug_img)

    # 제목 설정 및 눈금 제거
    axes[0].set_axis_off()
    axes[0].set_title('original')
    axes[1].set_axis_off()
    axes[1].set_title('augmentation')

    plt.show()

# show_augmentation_result()

# [Step3] CNN 모델 구조 구현
# Convolution Layer가 2개만 존재하는 CNN 구조 구현
# 각 이미지의 output 벡터 차원은 100차원으로 가정    
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1)
        self.fc = nn.Linear(4*4*20, 100) # fully connected layer
    
    def forward(self, x):
        x = F.relu(self.conv1(x)) # (batch, 1, 28, 28) -> (batch, 10, 24, 24)  (conv1의 kernel_size가 5이므로 기존 28*28사이즈에서 28-5+1을 계산하면 24*24가 됨)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (batch, 10, 24, 24) -> (batch, 10, 12, 12)
        x = F.relu(self.conv2(x)) # (batch, 10, 12, 12) -> (batch, 20, 8, 8)
        x = F.max_pool2d(x, kernel_size=2, stride=2) # (batch, 20, 8, 8) -> (batch, 20, 4, 4)
        x = x.view(-1, 4*4*20) # (batch, 20, 4, 4) -> (batch, 320)
        x = F.relu(self.fc(x)) # (batch,320) -> (batch, 100)
        return x # (batch, 100)
    

# [Step4] Loss 함수 구현
"""
SimCLR에서는 N개의 이미지로 구성된 배치에서 각 이미지별 augmentation된 N개의 이미지를 합쳐 총 2N개의 이미지를 최종 배치로 구성함
이후 (해당 이미지와 augmentation이미지) pair만 positive data(분자 부분)으로 간주하고
(해당 이미지와 나머지 이미지)의 2N-2개의 pair들은 negative data(분모 부분)으로 간주하여
contrastive loss를 계산함

GPU의 연산 효울성을 위해 배치 내 연산을 행렬 형태로 한번에 수행되게 만드는 것이 중요하며,
이 과정이 포함된 loss함수의 구현은 아래 사이트의 코드 이용

https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
""" 

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature # contrastive loss에서 tau값이며 하이퍼파라미터임

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    # loss 분모 부분의 negative sample 간의 내적 합만을 가져오기 위한 마스킹 행렬
    def mask_correlated_samples(self, batch_size):
        N = 2*batch_size
        """
         [
            N   N 
            N   N
         ]
         형태에서 각 N들의 대각선 요소들을 0으로 설정
        """
        mask = torch.ones((N,N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward(self, z_i, z_j):
        N = 2*self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        # z.unsqueeze(1)을 통해 1 x n형태로 z의 요소들이 할당되며, z.unsqueeze(0)을 통해 n x 1의 형태로 z의 요소들이 할당됨
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # loss 분자 부분의 원본과 augmentation 이미지 간의 내적 합을 가져오기 위한 부분
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim= 1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    

# [Step5] Training
    
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm # progress bar 출력을 위한 모듈 (jupyter notebook 사용시)
from tqdm import tqdm # progress bar 출력을 위한 모듈 (콘솔용)

X_train_aug = cutout_and_rotate(X_train) # 각 X_train 데이터에 대하여 augmentation
X_train_aug = X_train_aug.to(device) # 학습을 위하여 GPU에 선언

dataset = TensorDataset(X_train, X_train_aug) # augmentation된 데이터와 pair를 구성
batch_size = 32

dataloader = DataLoader(dataset, batch_size=batch_size)

model = CNN() # 모델 변수 선언
loss_func = SimCLR_Loss(batch_size, temperature=0.5) # loss 함수 선언

# train 코드 예시
epochs = 10
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(1, epochs + 1):
    total_loss = 0
    
    for data in tqdm(dataloader):
        origin_vec = model(data[0])
        aug_vec = model(data[1])

        loss = loss_func(origin_vec, aug_vec)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch : %d, Avg Loss : %.4f'%(i, total_loss / len(dataloader)))

# [Step6] 분류를 위한 다운스트림 모델 선언
# 위에서 학습한 CNN 구조의 모델에 class 개수만큼의 차원으로 projection을 진행하는 mlp layer를 장착하여 최종 class 분류를 위한 다운스트림 모델 선언
# 단일 mlp layer 만을 이용하여 projection하는 상황을 가정
class CNN_classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.CNN = model # contrastive learning으로 학습해둔 모델을 불러오기
        self.mlp = nn.Linear(100, 10) # class 차원 개수로 projection (CNN의 마지막 fully connected layer가 100차원 출력임. 100차원 출력값을 10개의 클래스로 분류)
    
    def forward(self, x):
        x = self.CNN(x) # (batch, 100)으로 변환
        x = self.mlp(x) # (batch, 10)으로 변환
        return x # (batch, 10)

# 학습을 위해 이미지와 라벨간의 pair를 이루어 dataloader를 선언
class_dataset = TensorDataset(X_train, y_train)
batch_size = 32

class_dataloader = DataLoader(class_dataset, batch_size=batch_size)

# [Step7] 분류 다운스트림 모델 학습 및 테스트
classifier = CNN_classifier(model).to(device) # 모델 선언, GPU 활용 지정
classifier_loss = nn.CrossEntropyLoss() # 분류를 위한 loss 함수

epochs = 10
classifier.train()

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

for i in range(1, epochs + 1):
    correct = 0
    for data in tqdm(class_dataloader):
        logits = classifier(data[0])

        loss = classifier_loss(logits, data[1].long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += torch.sum(torch.argmax(logits, 1) == data[1]).item() # 정확도 산출을 위하여 정답 개수 누적
    
    print('Epoch : %d, Train Accuracy : %.2f%%'%(i, correct * 100 / len(X_train)))


# [Step8] 테스트 데이터셋을 이용하여 검증
test_dataset = TensorDataset(X_test, y_test) # 테스트 데이터와 라벨 pair
batch_size = 32

test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
classifier.eval() # 테스트 모드로 전환

correct = 0
for data in tqdm(test_dataloader):
    logits = classifier(data[0])
    correct += torch.sum(torch.argmax(logits, 1) == data[1]).item() # 정확도 산출을 위하여 정답 개수 누적

print('Test Accuracy : %.2f%%'%(correct * 100 / len(X_test)))


