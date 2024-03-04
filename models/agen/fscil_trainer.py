from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
# from .Network import *
from .helper import *
from utils import *
from dataloader.data_utils import *
from CLIP import clip
import pickle
from losses import SupContrastive

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.model = MYNET(self.args, mode=self.args.base_mode)
        # self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_class_names(self, args):  # 获取类名
        class_names, class_attrs = [], []
        if args.dataset == 'car100':
            dataset_path = 'data/stanford_cars/car_name.txt'
            with open(dataset_path, "r") as file:
                for line in file:
                    name_str = line.strip().split(' ')[1]
                    class_name = name_str.replace('_', ' ')
                    class_names.append(class_name)
        if args.dataset == 'my_data':
            dataset_path = 'data/My_Data/mydata.txt'
            attr_path = 'data/My_Data/mydata_attr.txt'
            with open(dataset_path, "r") as file:
                for line in file:
                    name_str = line.strip()
                    class_name = name_str.replace('_', ' ')
                    class_names.append(class_name)
            with open(attr_path, "r") as file:
                for line in file:
                    attr_str = line.strip()
                    class_attr = attr_str.replace('_', ' ')
                    class_attrs.append(class_attr)
            return class_names, class_attrs
        if args.dataset == 'cub200':
            dataset_path = 'data/CUB_200_2011/classes.txt'
            attr_path = 'data/CUB_200_2011/class_attributes.txt'
            with open(dataset_path, "r") as file:
                for line in file:
                    name_str = line.strip().split('.')[1]
                    class_name = name_str.replace('_', ' ')
                    class_names.append(class_name)
            with open(attr_path, "r") as file:
                for line in file:
                    attr_str = line.strip()
                    class_attr = attr_str.replace('_', ' ')
                    class_attrs.append(class_attr)
            return class_names, class_attrs
        if args.dataset == 'aircraft100':
            dataset_path = 'data/fgvc-aircraft-2013b/variants.txt'
            family_path = 'data/fgvc-aircraft-2013b/variant_family.txt'
            with open(dataset_path, 'r') as f:
                for line in f:
                    list = line.strip('\n')
                    class_names.append(list)
            with open(family_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    class_attrs.append(line)
            return class_names, class_attrs
        elif args.dataset == 'cifar100':
            with open('data/cifar-100-python/meta', 'rb') as infile:
                data = pickle.load(infile, encoding='latin1')
                names = data['fine_label_names']
                for name in names:
                    class_name = name.replace('_', ' ')
                    class_names.append(class_name)

        elif args.dataset == 'mini_imagenet':
            imagenet_labels = {}
            file_path = 'data/miniimagenet/imagenet.txt'
            with open(file_path, "r") as file:
                for line in file:
                    list = line.split(':')
                    list[1] = list[1].strip().split(',')[0]
                    imagenet_labels[list[0]] = list[1]
            file_path = 'data/miniimagenet/split/test.csv'
            labels = [x.strip().split(',')[1] for x in open(file_path, 'r').readlines()][1:]
            for label in labels:
                class_name = imagenet_labels[label]
                if class_name not in class_names:
                    class_names.append(class_name)
        return class_names

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()
        if (args.dataset in ['mini_imagenet', 'cifar100']):
            class_names = self.get_class_names(args)
            print('text:a photo of a name')
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {c}") for c in class_names]).to(
                device)
        elif (args.dataset == 'cub200'):
            class_names, class_attrs = self.get_class_names(args)
            print('text:a photo of a name and attr')
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {n} and {a}") for n, a in zip(class_names, class_attrs)]).to(
                device)
        elif (args.dataset == 'aircraft100'):
            print('text:a photo of a name aircraft and it is a variant of family')
            class_names, family = self.get_class_names(args)
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {n} aircraft and it is a variant of {f}") for n, f in
                 zip(class_names, family)]).to(device)
        elif (args.dataset == 'my_data'):
            print('text:a photo of a name medicine and attr')
            class_names, class_attrs = self.get_class_names(args)
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {n} medicine and {a}") for n, a in
                 zip(class_names, class_attrs)]).to(device)
        # init train statistics
        result_list = [args]
        criterion = SupContrastive()
        criterion = criterion.cuda()
        for session in range(args.start_session, args.sessions):
            # for session in range(args.start_session, 1):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                train_set.multi_train = True
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, criterion, text_inputs, trainloader, optimizer, scheduler, epoch,
                                        args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)
                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    train_set.multi_train = False
                    self.model = replace_base_fc(train_set, text_inputs[:args.base_class], testloader.dataset.transform,
                                                 self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.update_fc(trainloader, text_inputs, np.unique(train_set.targets), session)
                loss, acc = test(self.model, testloader, 0, args, session, validation=False)
                tsl, tsa, best_epoch = loss, acc, -1
                # save model
                print('Replace the fc with average embedding')
                print('acc={} loss={}'.format(acc * 100, loss))
                self.best_model_dict = deepcopy(self.model.state_dict())
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None