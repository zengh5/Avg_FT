import torch
import numpy as np
from torchvision import models
from torchvision.models import Inception_V3_Weights, ResNet50_Weights, DenseNet121_Weights, VGG16_BN_Weights, swin_transformer
from torchvision import transforms
from PIL import Image

# customized networks: adding a few outputs to conventional networks
from net.resnet import ResNet18, ResNet50
from net.densenet import densenet121
from net.vgg16bn import vgg16_bn
from net.inception import inception_v3
# core functions
from attack_Ave_FT import AaFAttack
# routine functions
from utils_ImageNetCom import load_ground_truth, Normalize
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## S0: Load model
# Surrogates, modify it to the dir of the pretrained model in your computer
# Here we use res50 as the surrogate
checkpt_dir = 'C://Users/86188/.cache/torch/hub/checkpoints/'

# model = inception_v3(weights=None, transform_input=True)
# model.load_state_dict(torch.load(checkpt_dir + 'inception_v3_google-0cc3c7bd.pth'))

model = ResNet50(num_classes=1000)
model.load_state_dict(torch.load(checkpt_dir + 'resnet50-0676ba61.pth'))

# temp = torch.load(checkpt_dir + 'densenet121-a639ec97.pth')      # Densenet
# model = densenet121(weights=temp).eval()

# temp = torch.load(checkpt_dir + 'vgg16_bn-6c64b313.pth')
# model = vgg16_bn(weights=temp).eval()
########
model = model.to(device)
model.eval()

# victim models，
model_1 = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True).eval()
# model_2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
model_3 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
model_4 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).eval()
model_5 = swin_transformer.swin_t(weights=swin_transformer.Swin_T_Weights.IMAGENET1K_V1).eval()   # swin-tiny


for param in model_1.parameters():
    param.requires_grad = False
# for param in model_2.parameters():
#     param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False
for param in model_5.parameters():
    param.requires_grad = False

model_1.to(device)
# model_2.to(device)
model_3.to(device)
model_4.to(device)
model_5.to(device)

## S1: Load data and common transforms
img_size = 299
batch_size = 4
clean_path = 'E://Python/AE_transfer/Target/dataset/images/'  # clean images
adv_path = 'adv_imgs/logit/res50/'   # Your AEs

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

## S2: Running attack
pos = np.zeros(5)
pos_avg_ft = np.zeros(5)

torch.manual_seed(42)
for k in range(0, 2):
    if k % 1 == 0:
        print(k)
    #### 1. preparing data ####
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_adv = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    X_cln = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    for i in range(batch_size_cur):
        X_adv[i] = trn(Image.open(adv_path + image_id_list[k * batch_size + i] + '.png'))
        X_cln[i] = trn(Image.open(clean_path + image_id_list[k * batch_size + i] + '.png'))
    labels_ori = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()
    # predefined random-target scenario
    labels_tar = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()

    #### 2. Fine-tuning ####
    # X_cln: The original image.
    # X_adv: The adversarial example crafted with a baseline attack, e.g., CE, Logit.

    attack = AaFAttack(model=model, device=device, epsilon=16 / 255.)
    X_adv_ft = attack.perturb(X_cln, X_adv, labels_tar, labels_ori)

    #### 3. verify before fine-tune ####
    X_adv_norm = norm(X_adv).detach()

    output = model(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[0] = pos[0] + sum(predict_adv == labels_tar).cpu().numpy()

    output = model_1(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[1] = pos[1] + sum(predict_adv == labels_tar).cpu().numpy()

    output = model_3(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos[2] = pos[2] + sum(predict_adv == labels_tar).cpu().numpy()

    output = model_4(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    pos[3] = pos[3] + sum(predict_adv == labels_tar).cpu().numpy()

    output = model_5(X_adv_norm)
    predict_adv = torch.argmax(output, dim=1)
    pos[4] = pos[4] + sum(predict_adv == labels_tar).cpu().numpy()

    #### after averaging along fine-tune (AaF) ####
    X_adv_norm = norm(X_adv_ft).detach()

    output = model(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_avg_ft[0] = pos_avg_ft[0] + sum(predict_adv2 == labels_tar).cpu().numpy()

    output = model_1(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_avg_ft[1] = pos_avg_ft[1] + sum(predict_adv2 == labels_tar).cpu().numpy()

    output = model_3(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    print(output.gather(1, labels_tar.unsqueeze(1)).sum().data)
    pos_avg_ft[2] = pos_avg_ft[2] + sum(predict_adv2 == labels_tar).cpu().numpy()

    output = model_4(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    pos_avg_ft[3] = pos_avg_ft[3] + sum(predict_adv2 == labels_tar).cpu().numpy()

    output = model_5(X_adv_norm)
    predict_adv2 = torch.argmax(output, dim=1)
    pos_avg_ft[4] = pos_avg_ft[4] + sum(predict_adv2 == labels_tar).cpu().numpy()

print('Targeted attack success (baseline): ' + str(pos))
print('Targeted attack success (ours):     ' + str(pos_avg_ft))
Done = 1
