import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Simple(nn.Module):
    def __init__(self, backbone, hidden_units, shake, shake_config):
        super(Simple, self).__init__()
        if shake:
            self.feature = backbone(shake_config=shake_config)
        else:
            self.feature = backbone()
        self.hidden_units = hidden_units
        self.hidden_linear_1 = nn.Linear(self.feature.final_feat_dim, self.hidden_units)
        self.hidden_linear_2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.classifier = nn.Linear(self.hidden_units, 1)
        self.loss_func = nn.BCELoss()


    def forward_loss(self, x, y):
        x = Variable(x.cuda())
        y = Variable(y.type(torch.FloatTensor).cuda()).view(-1, 1)

        out = self.feature(x)
        out = F.dropout(F.relu(self.hidden_linear_1(out)))
        out = F.dropout(F.relu(self.hidden_linear_2(out)))
        out = torch.sigmoid(self.classifier(out))
        return out, self.loss_func(out, y)

    def train_loop(self, epoch, real_loader, rumor_loader, optimizer, scheduler, writer):
        print_freq = 10
        avg_loss = 0
        num_real_loader = len(real_loader) - 1
        num_rumor_loader = len(rumor_loader) - 1
        iter_real = iter(real_loader)
        iter_rumor = iter(rumor_loader)
        epoch_len = max(num_real_loader, num_rumor_loader)

        for i in range(epoch_len):
            if i % num_real_loader == 0:
                iter_real = iter(real_loader)
            if i % num_rumor_loader == 0:
                iter_rumor = iter(rumor_loader)

            real_image, real_label = iter_real.next()
            rumor_image, rumor_label = iter_rumor.next()
            x = torch.cat([real_image, rumor_image], 0)
            y = torch.cat([real_label, rumor_label], 0)

            optimizer.zero_grad()
            (_, loss) = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            writer.add_scalar('loss/total_loss', loss.item(), epoch * epoch_len + i)

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, epoch_len, avg_loss / float(i + 1)))

        if scheduler is not None:
            scheduler.step()
            print('Epoch:', epoch, 'LR:', scheduler.get_lr())

    def val_loop(self, epoch, real_loader, rumor_loader, writer):
        num_real_loader = len(real_loader) - 1
        num_rumor_loader = len(rumor_loader) - 1
        iter_real = iter(real_loader)
        iter_rumor = iter(rumor_loader)
        epoch_len = max(num_real_loader, num_rumor_loader) + 1
        acc = []

        for i in range(epoch_len):
            if i % num_real_loader == 0:
                iter_real = iter(real_loader)
            if i % num_rumor_loader == 0:
                iter_rumor = iter(rumor_loader)

            real_image, real_label = iter_real.next()
            rumor_image, rumor_label = iter_rumor.next()
            x = torch.cat([real_image, rumor_image], 0)
            y = torch.cat([real_label, rumor_label], 0)

            (score, _) = self.forward_loss(x, y)

            acc = 100 - np.average(np.abs((score > 0.5).int().squeeze().cpu() - y.type(torch.IntTensor))) * 100

        acc = np.mean(acc)
        writer.add_scalar('acc/val_acc', acc.item(), epoch)
        return acc

    def test_loop(self, novel_loader):
        acc = []

        for i, (x, y) in enumerate(novel_loader):
            (score, _) = self.forward_loss(x, y)
            acc.append(100 - np.average(np.abs((score > 0.5).int().squeeze().cpu() - y.type(torch.IntTensor))) * 100)

        acc = np.mean(acc)
        return acc
