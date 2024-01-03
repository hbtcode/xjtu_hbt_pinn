# (h3*P.x).x+(h3*P.y).y=1/2*h.x
# h(x)=ax+b
# p(0,y)=0
# p(x,0)=0
# p(20,y)=0
# p(x,20)=0
import os.path
import time

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
# 这个888888确定了一个随机数的初始状态
setup_seed(888888)
N = 100


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# 这部分是计算预测出来的u对于给定变量的梯度或者求导
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def interior(n=N):
    x = torch.rand(n, 1)
    cond = 0.5 * a * torch.ones_like(x)
    return x.requires_grad_(True), cond


def left(n=N):
    x = torch.zeros(n, 1)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), cond


def right(n=N):
    x = torch.ones(n, 1)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), cond


def loss_function_interior(p):
    # 损失函数L1，方程结构损失
    x, cond = interior()
    px = x
    h = (a * x + b)
    return loss(gradients(gradients(px, x, 1) * h * h * h, x, 1),
                cond)


def loss_function_left(p):
    # 损失函数L2，下边界损失
    x,cond = left()
    px = x
    return loss(px, cond)


def loss_function_right(p):
    # 损失函数L3，下边界损失
    x, cond = right()
    px = x
    return loss(px, cond)


# 根据两点来计算该直线
def ab_from_points(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


if __name__ == "__main__":
    # 训练迭代次数
    epochs = 10000;
    # 边界和定义域配置数据点个数
    h = 1000
    h1 = input("请输入第一个h1坐标点（格式为 x,y）：")
    x1, y1 = map(float, h1.split(','))
    h2 = input("请输入第二个h2坐标点（格式为 x,y）：")
    x2, y2 = map(float, h2.split(','))
    a, b = ab_from_points(x1, y1, x2, y2)
    # x1 = 20
    # x2 = 0
    # for h1 in range(5):
    #     a, b = ab_from_points(x1, h1 + 1, x2, h1 + 101)
    filename = 'parameters2/model_parameters(a=' + str(a) + '_b=' + str(b) + ').pth'  # 文件路径
    p = Network()
    loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(params=p.parameters())
    count = 0
    start_time = time.time()
    if os.path.exists(filename):
        p.load_state_dict(torch.load(filename))
        count = 1
    else:
        # loss_history = []
        for i in range(epochs):
            opt.zero_grad()
            l = loss_function_interior(p)  + loss_function_right(p) \
                + loss_function_left(p)
            l.backward()
            opt.step()
            # loss_history.append(l.item())
            if i % 100 == 0:
                print("第" + str(h1) + "点的第" + str(i) + "次")

        torch.save(p.state_dict(), filename)

    # plt.plot(loss_history)
    # plt.title('Loss variation over increasing epochs')
    # plt.xlabel('epochs')
    # plt.ylabel('loss value')
    # plt.show()
    # plt.savefig("img/PINN loss_epochs.png")

    xc = torch.linspace(0, 20, h)
    xm, ym = torch.meshgrid(xc, xc)
    xx = xm.reshape(-1, 1)
    yy = ym.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    p_pred = p(xy)
    u_pred_fig = p_pred.reshape(h, h)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig.detach().numpy())
    ax.text2D(0.5, 0.9, "PINN", transform=ax.transAxes)
    plt.show()
    if count == 0:
        fig.savefig("img2/PINN solve a=" + str(a) + "_ b=" + str(b) + ".png")
    end_time = time.time()
    print("共用时" + str(end_time - start_time) + "秒")
