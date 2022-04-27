import time

from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings
import numpy as np

NUM_EPOCHS = 20 # 20

print('Device:',d2l.try_gpu())

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = '/scratch/data/datasets/pokemon/pokemon'
pokemon = torchvision.datasets.ImageFolder(data_dir)

batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())

warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[0:20,:,:,:].permute(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break


class G_block(nn.Module):
    def __init__(self, channels, nz=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(
            nz, channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk(x).shape


def Conv2DTranspose(channels, kernel_size, strides, padding, use_bias, nc=3):
    return nn.ConvTranspose2d(nc, channels, kernel_size=kernel_size,stride=strides, padding=padding, bias=use_bias)

n_G = 64
net_G = nn.Sequential(
    G_block(n_G*8, nz=100, strides=1, padding=0),  # Output: (64 * 8, 4, 4)
    G_block(n_G*4, n_G*8),  # Output: (64 * 4, 8, 8)
    G_block(n_G*2, n_G*4),  # Output: (64 * 2, 16, 16)
    G_block(n_G, n_G*2),    # Output: (64, 32, 32)
    Conv2DTranspose(
              3, nc=n_G, kernel_size=4, strides=2, padding=1, use_bias=False),
    nn.Tanh())              # Output: (3, 64, 64)


x = torch.zeros((1, 100, 1, 1))
print('net_G(x).shape:',net_G(x).shape)


alphas = [0, 0.2, 0.4, .6, .8, 1]
x = torch.arange(-2, 1, 0.1)
Y = [nn.LeakyReLU(alpha)(x).numpy() for alpha in alphas]
d2l.plot(d2l.numpy(x), Y, 'x', 'y', alphas)

class D_block(nn.Module):
    def __init__(self, channels, nc=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(
            nc, channels, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk(x).shape


def Conv2D(channels, kernel_size, use_bias, nc=3):
    return nn.Conv2d(nc, channels, kernel_size=kernel_size, bias=use_bias)


n_D = 64
net_D = nn.Sequential(
    D_block(n_D),    # Output: (64, 32, 32)
    D_block(n_D*2, n_D),  # Output: (64 * 2, 16, 16)
    D_block(n_D*4, n_D*2),  # Output: (64 * 4, 8, 8)
    D_block(n_D*8, n_D*4),  # Output: (64 * 8, 4, 4)
    Conv2D(1, nc=n_D*8, kernel_size=4, use_bias=False))  # Output: (1, 1, 1)

x = torch.zeros((1, 3, 64, 64))
net_D(x).shape


def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size, 1, 1, 1), device=X.device)
    zeros = torch.zeros((batch_size, 1, 1, 1), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D

def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,1,1,1), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step()
    return loss_G

def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    losses = []
    overall_timer =d2l.Timer()
    for epoch in range(1, num_epochs + 1):
        print('Epoch:',epoch)
        # Train one epoch
        epoch_timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for index, (X, _) in enumerate(data_iter):
            print('Processing batch: ',epoch,'-',index,sep='')
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
            loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
            print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
                  f'{metric[2] / epoch_timer.stop():.1f} examples/sec on {str(device)} ('
                  f'{metric[2]} examples in {epoch_timer.times[-1]:.1f} seconds during this epoch)')
        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
        losses.append((epoch, loss_D, loss_G))
    end = overall_timer.stop()
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{end / num_epochs:.1f} sec/epoch on {str(device)}')
    print(f'Num epochs: {num_epochs}')
    print(f'Total training time: {end:.1f} seconds')
    print('Started at:',time.strftime("%a, %d %b %Y %H:%M:%S local", time.localtime(
        overall_timer.tik)))
    print('Ended at:', time.strftime("%a, %d %b %Y %H:%M:%S local", time.localtime(
        overall_timer.tik+end)))
    return losses


latent_dim, lr, num_epochs = 100, 0.005, NUM_EPOCHS
device = d2l.try_gpu()
losses = train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,device=device)

Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1)).to(device)

fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
fake_cpu = fake_x.cpu()
imgs = np.concatenate(
             [np.concatenate([fake_cpu[i * 7 + j].detach().numpy() for j in range(7)], axis=1)
              for i in range(len(fake_cpu)//7)], axis=0)

np.save('imgs.npy',imgs)
losses = np.array(losses)
np.save('losses.npy',losses)

# plt.imshow(imgs)
# plt.plot(losses[:,0],losses[:,1:3])
