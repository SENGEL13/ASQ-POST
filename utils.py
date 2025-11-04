from tqdm import tqdm, trange
import time
import torch
import logging
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import shutil
import os
import yaml
import re
import csv

class MetricsLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self._initialize_csv()

    def _initialize_csv(self):
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'top1', 'top5'])

    def log_metrics(self, epoch, top1, top5):
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, top1, top5])

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)
        
def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def test(model, test_loader, gpu):
    """
    Get the test performance
    """
    #start = time.time()
    model.eval()
    correct = 0
    total_num = 0
    with tqdm(total=len(test_loader.dataset)) as progressbar:
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(gpu), target.cuda(gpu)
            output = model(data)
            pred = output.data.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
            progressbar.set_postfix(acc=100. * correct / total_num)
            progressbar.update(target.size(0))
            #print('testing_correct: ', correct / total_num, '\n')
    #end = time.time()
    return correct / total_num

def update_tempratures(model):
    count = 0
    current_temprature = 0
    for name, m in model.named_modules():
        if name[-9:]=="attention":
            m.update_temprature()
            if not count:
                current_temprature = m.temprature
            count+=1
    if count:
        if current_temprature==1:
            return False, current_temprature
        else:
            return True, current_temprature
    else:
        return False, None

def train_vm(model, train_loader, test_loader, device, num_epochs=90):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_acc = 0
    last_acc = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        model.train()
        total_data = 0
        total_loss = 0
        with tqdm(total=len(train_loader.dataset)) as progressbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_data += data.size(0)
                progressbar.set_postfix(loss=total_loss/total_data) 
                progressbar.update(target.size(0))
        scheduler.step()
        acc = test(model, test_loader, device=device)
        last_acc = acc
        if acc > best_acc:
            best_acc = acc
        print(f"Current acc: {acc}")
    return best_acc, last_acc


def train(train_loader, model, criterion, optimizer, epoch, gpu, print_freq=10, gpu_info=None):#args
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    start = time.time()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    # print('learning_rate:', cur_lr)
    logging.info(f"{gpu_info}=> learning_rate: {cur_lr}.")
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i, start, gpu_info)

def validate(val_loader, model, criterion, gpu, print_freq=10, gpu_info=None):#save_path
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i, start, gpu_info)

        logging.info(gpu_info+' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def fmt_time(time_dif):
    total_seconds = int(time_dif)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.num_batches = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, start_time, gpu_info=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries.append(f"Spent(Left) {fmt_time(time.time()-start_time)}({fmt_time((self.num_batches-batch-1)*self.meters[0].avg)})") 
        logging.info(gpu_info+'\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res