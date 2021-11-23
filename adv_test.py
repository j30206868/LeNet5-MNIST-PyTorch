import cv2
import torch
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def imshow(img):
    img = img.cpu().detach().numpy()
    cv2.namedWindow('img', 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyWindow('img')  

def custom_fast_gradient(model_fn, x, eps, clip_min, clip_max, y):
    x = x.clone().detach().requires_grad_(True)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y.long())
    loss.backward()
    # with torch.no_grad():
    optimal_perturbation = eps * torch.sign(x.grad)
    adv_x = x + optimal_perturbation
    adv_x = torch.clamp(adv_x, clip_min, clip_max)
    
    return x + optimal_perturbation
if __name__ == '__main__':
    batch_size = 1
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    ### load model
    model = torch.load('models/mnist_0.99.pkl')

    train_x, train_label = next(train_loader.__iter__())

    adv_x = train_x.clone().detach()
    eps = 1.0/255.0
    for step in range(40):
        adv_x = custom_fast_gradient(model, adv_x, eps, 0, 1, train_label)
        ### Use cleverhans fast_gradient_method
        # from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
        # adv_x = fast_gradient_method(model, train_x, eps, np.inf, 0, 1, train_label)

    noise = adv_x - train_x
    softmax = torch.nn.Softmax(dim=-1)
    print('noise_output[0] = '+str(softmax(model(adv_x)[0])))
    print('output[0] = '+str(softmax(model(train_x)[0])))
    print('sum(abs(noise)) = '+str(torch.sum(torch.abs(noise))))
    imshow((adv_x)[0,0])
