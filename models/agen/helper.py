# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from losses import SupContrastive
from random import choice


# import cv2
# import numpy as np


def base_train(model, criterion, text, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_ for _ in batch]
        b, c, h, w = data[0].shape
        original = data[0].cuda(non_blocking=True)
        data[1] = data[1].cuda(non_blocking=True)
        data[2] = data[2].cuda(non_blocking=True)
        train_label = train_label.cuda(non_blocking=True)
        data_classify = transform1(original)
        data_query = transform1(data[1])
        data_key = transform1(data[2])

        # data_classify = original
        # data_query = data[1]
        # data_key = data[2]

        label1 = torch.cat((train_label * 2, train_label * 2 + 1))
        # label2 = torch.cat((train_label,train_label))
        label2 = train_label
        label = torch.eq(label1.unsqueeze(0), label1.unsqueeze(1)).float()
        logits, x1, text_logits = model(im_cla=data_classify, im_q=data_query, im_k=data_key, text_inputs=text)
        # logits,text_logits = model(im_cla=data_classify,text_inputs=text)

        logits = logits[:, :args.base_class * 2]
        text_logits = text_logits[:, :args.base_class]

        loss1 = criterion(x1, label)
        total_loss = F.cross_entropy(logits, label1) + F.cross_entropy(text_logits, label2) * args.beta + loss1
        logits = (logits[:b, ::2] + logits[b:, 1::2]) / 2
        acc = count_acc(logits, train_label)
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def transform1(images):
    k = 1
    trans_image = torch.rot90(images, k, (2, 3))
    return torch.cat((images, trans_image))


def replace_base_fc(trainset, text_inputs, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            data = transform1(data)
            trans_label = label * 2 + 1
            label = label * 2
            label = torch.cat((label, trans_label))
            text = text_inputs
            embedding, text_feature = model(im_cla=data, text_inputs=text)
            # embedding = model(im_cla=data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    text_feature = text_feature.cpu()
    for class_index in range(args.base_class * 2):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        embedding_this = args.alpha * embedding_this + (1 - args.alpha) * text_feature[class_index // 2]
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)
    print(proto_list.shape, type(proto_list))
    model.fc.weight.data[:args.base_class * 2] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    # num = args.num_represent
    # attention = attention.eval()
    vl = Averager()
    va = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.shape[0]
            data = transform1(data)
            logits = model(data)
            logits = logits[:, :test_class * 2]
            logits = (logits[:b, ::2] + logits[b:, 1::2]) / 2
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])

        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
    return vl, va