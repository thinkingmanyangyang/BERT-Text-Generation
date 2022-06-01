import torch
from torch import nn
import torch.nn.functional as F

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1)
        return loss


class LabelSmoothing2(nn.Module):
    "Implement label smoothing.  size表示类别总数  "

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing2, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # if i=y的公式
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        size = x.size(1)
        true_dist = x.data.clone()  # 先深复制过来
        # print true_dist
        true_dist.fill_(self.smoothing / (size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        self.true_dist = true_dist
        return self.criterion(x, true_dist, requires_grad=False)


if __name__ == '__main__':
    loss_fct = LabelSmoothLoss(smoothing=0.05)
    loss_fct2 = LabelSmoothing2(smoothing=0.05)
    input = torch.tensor([[1, -1, -1], [-100, 100, -100], [100, -100, -100]], dtype=torch.float)
    target = torch.tensor([0, 1, 0])
    loss = loss_fct(input, target)
    print(loss)
    loss = loss_fct(input, target)
    print(loss)