import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

from utils_ImageNetCom import FIAloss, DI_keepresolution, gkern, Normalize

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()


class AaFAttack(object):
    # ImageNet
    def __init__(self, model=None, device=None, epsilon=16 / 255., k=10, alpha=1.6 / 255., prob=0.7,
                 mask_num=30, mu=1.0, model_name='res18', decay_factor=0.8, k_wu=5):
        # set Parameters
        self.model = model.to(device)
        self.epsilon = epsilon
        self.k = k + k_wu
        self.k_wu = k_wu
        self.alpha = alpha
        self.prob = prob          # for normal model, drop 0.3; for defense models, drop 0.1
        self.mask_num = mask_num  # according to RPA paper: 30
        self.mu = mu
        self.device = device
        self.model_name = model_name
        self.decay_factor = decay_factor

    def perturb(self, X_nat, X_mid, y, y_ori):
        '''
        :param X_nat: the original image
        :param X_mid: AE crafted with a baseline attack, e.g., CE, Logit.
        :param y: target label, e.g., 'tench'.
        :param y_ori: original label
        :return: X_adv_AaF, AE fine-tuned with the proposed AaF method
        '''
        self.alpha = self.epsilon / 16.                            # the step size of Fine-tune

        # get grads
        labels = y.clone().detach().to(self.device)
        labels_ori = y_ori.clone().detach().to(self.device)
        _, temp_x_l1, temp_x_l2, temp_x_l3, temp_x_l4 = self.model.features_grad_multi_layers(X_nat)

        batch_size = X_nat.shape[0]
        image_size = X_nat.shape[-1]

        # calculate the feature importance (y_o) from the clean image
        # grad_sum_l1 = torch.zeros(temp_x_l1.shape).to(device)
        # grad_sum_l2 = torch.zeros(temp_x_l2.shape).to(device)
        grad_sum_l3 = torch.zeros(temp_x_l3.shape).to(device)
        # grad_sum_l4 = torch.zeros(temp_x_l4.shape).to(device)
        for i in range(self.mask_num):
            self.model.zero_grad()
            ######## RPA
            img_temp_i = X_nat.clone()
            if i % 4 == 0:
                mask1 = np.random.binomial(1, 0.7, size=(batch_size, image_size, image_size, 3))
                mask2 = np.random.uniform(0, 1, size=(batch_size, image_size, image_size, 3))
                mask = np.where(mask1 == 1, 1, mask2)
            elif i % 4 == 1:
                mask = patch_by_strides((batch_size, image_size, image_size, 3), (3, 3), 0.7)
            elif i % 4 == 2:
                mask = patch_by_strides((batch_size, image_size, image_size, 3), (5, 5), 0.7)
            else:
                mask = patch_by_strides((batch_size, image_size, image_size, 3), (7, 7), 0.7)
            mask = torch.tensor(mask.transpose(0, 3, 1, 2), dtype=torch.float32).cuda()
            img_temp_i = norm(img_temp_i * mask)
            ########
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            logit_label = logits.gather(1, labels_ori.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            # grad_sum_l1 += x_l1.grad
            # grad_sum_l2 += x_l2.grad
            grad_sum_l3 += x_l3.grad
            # grad_sum_l4 += x_l4.grad

        chan_num = grad_sum_l3.shape[1]
        gaussian_kerneln = np.expand_dims(np.repeat(np.stack([kernel]), repeats=chan_num, axis=0), 1)
        gaussian_kerneln = torch.from_numpy(gaussian_kerneln).cuda()
        grad_sum_l3 = F.conv2d(grad_sum_l3, gaussian_kerneln, bias=None, stride=1, padding=(2, 2), groups=chan_num)
        # avr
        grad_sum_l3 = grad_sum_l3 / grad_sum_l3.std()

        Sec = 1
        # calculate the feature importance (y_t) from a mid image
        # grad_sum_mid_l1 = torch.zeros(temp_x_l1.shape).to(device)
        # grad_sum_mid_l2 = torch.zeros(temp_x_l2.shape).to(device)
        grad_sum_mid_l3 = torch.zeros(temp_x_l3.shape).to(device)
        # grad_sum_mid_l4 = torch.zeros(temp_x_l4.shape).to(device)
        for i in range(self.mask_num):
            self.model.zero_grad()  # Hui Zeng
            ######## RPA
            img_temp_i = X_mid.clone()
            if i % 4 == 0:
                mask1 = np.random.binomial(1, 0.7, size=(batch_size, image_size, image_size, 3))
                mask2 = np.random.uniform(0, 1, size=(batch_size, image_size, image_size, 3))
                mask = np.where(mask1 == 1, 1, mask2)
            elif i % 4 == 1:
                mask = patch_by_strides((batch_size, image_size, image_size, 3), (3, 3), 0.7)
            elif i % 4 == 2:
                mask = patch_by_strides((batch_size, image_size, image_size, 3), (5, 5), 0.7)
            else:
                mask = patch_by_strides((batch_size, image_size, image_size, 3), (7, 7), 0.7)
            mask = torch.tensor(mask.transpose(0, 3, 1, 2), dtype=torch.float32).cuda()
            img_temp_i = norm(img_temp_i * mask)
            #############
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            logit_label = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            # grad_sum_mid_l1 += x_l1.grad
            # grad_sum_mid_l2 += x_l2.grad
            grad_sum_mid_l3 += x_l3.grad
            # grad_sum_mid_l4 += x_l4.grad

        grad_sum_mid_l3 = F.conv2d(grad_sum_mid_l3, gaussian_kerneln, bias=None, stride=1, padding=(2, 2),
                                   groups=chan_num)
        # avr
        grad_sum_mid_l3 = grad_sum_mid_l3 / grad_sum_mid_l3.std()
        ########### combined aggregate gradient
        beta = 0.2
        grad_sum_new_l3 = grad_sum_mid_l3 - beta * grad_sum_l3

        # initialization
        g = 0
        x_cle = X_nat.detach()
        x_adv_2 = X_mid.clone()
        eta_prev = 0
        accumaleted_factor = 0

        for epoch in range(self.k):
            self.model.zero_grad()
            x_adv_2.requires_grad_()
            x_adv_2_DI = DI_keepresolution(x_adv_2)  # DI
            x_adv_norm = norm(x_adv_2_DI)  # [0, 1] to [-1, 1]

            _, _, mid_feature_l3, _ = self.model.multi_layer_features(x_adv_norm)
            loss = FIAloss(grad_sum_new_l3, mid_feature_l3)  # FIA loss
            loss.backward()

            grad_c = x_adv_2.grad
            # TI, MI
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            g = self.mu * g + grad_c

            # update AE
            x_adv_2 = x_adv_2 + self.alpha * g.sign()
            with torch.no_grad():
                eta = x_adv_2 - x_cle

                ## Core code of AaF
                # memory-efficient implementaion of:
                # x_0*decay_factor^(epoch) + x_1*decay_factor^(epoch-1) + ...+ x_(epoch)
                if epoch >= self.k_wu:         # 5 (10 for CE) iterations for warming up
                    eta = (eta + accumaleted_factor * eta_prev) / (accumaleted_factor + 1)
                    accumaleted_factor = self.decay_factor * (accumaleted_factor + 1)
                    eta_prev = eta

                eta = torch.clamp(eta, min=-self.epsilon, max=self.epsilon)
                X_adv_AaF = torch.clamp(x_cle + eta, min=0, max=1).detach_()

            x_adv_2 = torch.clamp(x_cle + eta, min=0, max=1)

        return X_adv_AaF


# patch_by_strides() is borrowed from RPA paper
def patch_by_strides(img_shape, patch_size, prob):
    X_mask = np.ones(img_shape)
    N0, H0, W0, C0 = X_mask.shape
    ph = H0 // patch_size[0]
    pw = W0 // patch_size[1]
    X = X_mask[:, :ph * patch_size[0], :pw * patch_size[1]]
    N, H, W, C = X.shape
    shape = (N, ph, pw, patch_size[0], patch_size[1], C)
    strides = (X.strides[0], X.strides[1] * patch_size[0], X.strides[2] * patch_size[0], *X.strides[1:])
    mask_patchs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
    ran_num = int(mask_len * (1 - prob))
    rand_list = np.random.choice(mask_len, ran_num, replace=False)
    for i in range(mask_patchs.shape[1]):
        for j in range(mask_patchs.shape[2]):
            for k in range(mask_patchs.shape[-1]):
                if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                    mask_patchs[:, i, j, :, :, k] = np.random.uniform(0, 1,
                                                                      (N, mask_patchs.shape[3], mask_patchs.shape[4]))
    img2 = np.concatenate(mask_patchs, axis=0, )
    img2 = np.concatenate(img2, axis=1)
    img2 = np.concatenate(img2, axis=1)
    img2 = img2.reshape((N, H, W, C))
    X_mask[:, :ph * patch_size[0], :pw * patch_size[1]] = img2
    return X_mask
