import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
from tqdm import tqdm
from torch.utils import data
import model
from torch.autograd import Variable
from visdom import Visdom


class Config(object):
    data_path = 'data/'  # 数据集存放的路径
    num_workers = 4  # 多进程加载数据的进程数
    image_size = 96  # 图像大小
    batch_size = 16
    max_epoch = 6000
    lr1 = 1E-4  # 生成器学习率
    lr2 = 1E-4  # 判别器学习率
    beta1 = 0.5  # Adam优化器的beta参数
    use_gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ndf = 64  # 判别器feature map数
    ngf = 64  # 生成器feature map数

    save_path = 'images/'  # 生成图片保存路径

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 100  # 每次间隔20个batch，visdom画图一次

    debug_file = 'temp/debug_gan'  # 存在该文件则进入debug模式
    d_every = 2  # 每1个batch训练一次判别器
    g_every = 1  # 每5个batch训练一次生成器
    decay_every = 200  # 每10个batch保存一次模型
    d_net_path = 'checkpoints/d_net'  # 判别器模型保存位置
    g_net_path = 'checkpoints/g_net'

    # 测试用的参数
    gen_img = 'result.png'
    # 从512张生成的图片保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


def train():
    loss_d, loss_g = 0, 0
    for ii, (img, _) in enumerate(dataloader):
        real_img = img
        if opt.use_gpu:
            real_img = real_img.to(device)
        if (ii + 1) % opt.d_every == 0:
            # print('训练判别器')
            optimizer_d.zero_grad()
            output = net_d(real_img)
            error_d_real = criterion(output, true_labels)
            loss_d += error_d_real.item()
            error_d_real.backward()

            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = net_g(noises).detach()
            fake_output = net_d(fake_img)
            error_d_fake = criterion(fake_output, fake_labels)
            loss_d += error_d_real.item()
            error_d_fake.backward()
            optimizer_d.step()

        if (ii + 1) % opt.g_every == 0:
            # print('训练生成器')
            optimizer_g.zero_grad()
            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = net_g(noises)
            fake_output = net_d(fake_img)
            error_g = criterion(fake_output, true_labels)
            loss_g += error_g.item()
            error_g.backward()
            optimizer_g.step()
    # print(f'loss_g:{loss_g}, loss_d:{loss_d}')
    loss_d_list.append(loss_d)
    loss_g_list.append(loss_g)


def visualization():
    vis = Visdom()
    fix_fake_imgs = net_g(fix_noises)
    normal_img = fix_fake_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5
    # print(normal_img,normal_img.shape)
    vis.images(normal_img, win='fixfake')


def generate_image():
    net_g, net_d = model.NetG(opt).eval(), model.NetD(opt).eval()
    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = Variable(noises)

    # 加载模型
    net_d = torch.load(opt.d_net_path)
    net_g = torch.load(opt.g_net_path)

    if opt.use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net_d.to(device)
        net_g.to(device)
        noises = noises.to(device)

    # 生成图片，计算分数
    fake_img = net_g(noises)
    score = net_d(fake_img).data

    indexes = score.topk(opt.gen_num)[1]
    result = []
    for ii in indexes:
        result.append(fake_img.data[ii])
    torchvision.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


def save_result(j=1):
    net_g, net_d = model.NetG(opt).eval(), model.NetD(opt).eval()
    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = Variable(noises)

    # 加载模型
    d_name = opt.d_net_path + '_' + str(i + 1) + '.pt'
    g_name = opt.g_net_path + '_' + str(i + 1) + '.pt'
    net_d = torch.load(d_name)
    net_g = torch.load(g_name)

    if opt.use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net_d.to(device)
        net_g.to(device)
        noises = noises.to(device)

    # 生成图片，计算分数
    fake_img = net_g(noises)
    score = net_d(fake_img).data

    indexes = score.topk(opt.gen_num)[1]
    result = []
    for ii in indexes:
        result.append(fake_img.data[ii])
    jpg_name = opt.save_path + str(j) + '.jpg'
    torchvision.utils.save_image(torch.stack(result), jpg_name, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    # 超参数加载
    opt = Config()

    # 数据初始化和加载
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.image_size),
        torchvision.transforms.CenterCrop(opt.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers,
                                 drop_last=True  # 防止batch不合适
                                 )

    # 定义加载网络
    map_location = lambda storage, loc: storage
    net_g = model.NetG(opt)
    net_d = model.NetD(opt)
    # try:
    #     print('正在加载模型...')
    #     net_g = torch.load(opt.g_net_path)
    #     net_d = torch.load(opt.d_net_path)
    #     print('加载模型成功')
    # except FileNotFoundError:
    #     print('无可用模型')

    # 定义优化器和损失
    optimizer_g = torch.optim.Adam(net_g.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(net_d.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = torch.nn.BCELoss()

    true_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))
    fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
    noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))

    if opt.use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net_d.to(device)
        net_g.to(device)
        criterion.to(device)
        true_labels, fake_labels = true_labels.to(device), fake_labels.to(device)
        fix_noises, noises = fix_noises.to(device), noises.to(device)

    loss_d_list = []
    loss_g_list = []
    with tqdm(total=opt.max_epoch) as t:
        for i in range(opt.max_epoch):
            train()
            if (i + 1) % opt.decay_every == 0:
                # print('保存模型中...')
                d_name = opt.d_net_path + '_' + str(i + 1) + '.pt'
                g_name = opt.g_net_path + '_' + str(i + 1) + '.pt'
                torch.save(net_d, d_name)
                torch.save(net_g, g_name)
                # print('保存模型成功')
            if (i + 1) % opt.plot_every == 0:
                visualization()
            if (i + 1) % opt.decay_every == 0:
                save_result(i + 1)
            t.set_postfix(d_loss=loss_d_list[-1], g_loss=loss_g_list[-1])
            t.update(1)

    plt.plot(range(1, opt.max_epoch + 1), loss_d_list, '-', label='loss_d')
    plt.plot(range(1, opt.max_epoch + 1), loss_g_list, '--', label='loss_g')
    plt.legend()
    plt.show()

    # generate_image()

    visualization()
