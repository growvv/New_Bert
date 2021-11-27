#训练loss可视化
import matplotlib.pyplot as plt


def draw_loss(loss_collect, name = 'loss.png', line=False):
    plt.figure(figsize=(12,8))
    if line:
        plt.plot(range(len(loss_collect)), loss_collect)
    else:
        plt.plot(range(len(loss_collect)), loss_collect, 'g.')
    plt.grid(True)
    plt.savefig('./result' + name)
    # plt.show()

if __name__ == "__main__":
    loss_collect = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    draw_loss(loss_collect, line=True)

