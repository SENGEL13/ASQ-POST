from pytorchcv.model_provider import get_model as ptcv_get_model
from q_resnet import *
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageNet
from utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import datetime
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torchvision.models import resnet18, resnet50

def ddp_setup(rank, world_size, gpu_ids):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_ids[rank])

def main(rank: int, world_size: int, gpu: list, cfg):
    distributed = world_size > 1
    # gpu = cfg["gpu"]
    # gpu_ids = list(map(int, gpu.split(",")))
    if distributed:
        ddp_setup(rank, world_size, gpu)
        gpu_id = gpu[rank]
    else:
        gpu_id = gpu

    arch = cfg["arch"]
    bit = cfg["bit"]
    resume = cfg["resume"]
    lr = cfg["lr"]
    save_path = cfg["save_path"]
    quant = cfg["quant"]
    if quant: 
        quant_arch = arch+"_{}".format(bit)
        save_path = save_path+quant_arch+"/"
    else:
        save_path = save_path+arch+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = MetricsLogger(save_path+'result.csv')
    # save yaml file
    with open(save_path+"default.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg,f,allow_unicode=True)
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=save_path + 'log.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    if distributed:
        gpu_info = f"GPU[{gpu_id}] "
    else:
        gpu_info = ""

    logging.info(f"{gpu_info}=> Start importing data.")
    
    data_path = cfg["data_path"]
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    train_batch_size = cfg["train_batch_size"]
    num_workers = cfg["num_workers"]
    train_resolution = 224
    crop_scale = 0.08
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(train_resolution, scale=(crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    data_percentage = cfg["data_percentage"]
    if data_percentage==1:
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    else:
        dataset_length = int(len(train_dataset) * data_percentage)
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset,[dataset_length, len(train_dataset) - dataset_length])
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(partial_train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            partial_train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    
    val_batch_size = cfg["val_batch_size"]
    test_resolution = (256, 224)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(test_resolution[0]),
            transforms.CenterCrop(test_resolution[1]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    logging.info(f"{gpu_info}=> Data import completed.")
    cnn = eval(arch)(pretrained=cfg["pretrained"]) #ptcv_get_model(arch, pretrained=cfg["pretrained"])
    if cfg["local_model"] is not None:
        logging.info(f"{gpu_info}=> Using Local Pretrained Model.")
        model_path = cfg["local_model"]
        loc = 'cuda:{}'.format(gpu_id)
        model_parameter = torch.load(model_path, map_location=loc)
        cnn.load_state_dict(model_parameter['state_dict'], strict=False)
    if quant:
        logging.info(f"{gpu_info}=> Model Initialization Begins.")
        model = eval("q_"+arch)(cnn, bit=bit)#.cuda()
        #model = torch.nn.DataParallel(cnn).to(device)
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data, target
            break
        init_out = model(data)
        logging.info(f"{gpu_info}=> Model initialization complete.")
    else:
        model = cnn

    if distributed:
        model.cuda(gpu_id)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
        #logging.info(f"{gpu_info}=> Run by using DDP.")
    elif gpu_id is not None:
        model = model.cuda(gpu_id)
        logging.info("=> Run by using gpu {}.".format(gpu_id))
    else:
        model = torch.nn.DataParallel(model).cuda()
        logging.info("=> Run by using DP.")
    
    
    best_epoch = 0
    start_epoch = 0
    best_acc1 = 0
    num_epochs = cfg["num_epochs"]
    momentum = cfg["momentum"]
    weight_decay = cfg["weight_decay"]
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    if resume:
        checkpoint_path = save_path + "checkpoint.pth.tar"
        if os.path.isfile(checkpoint_path):
            if gpu_id is None:
                checkpoint = torch.load(checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu_id)
                checkpoint = torch.load(checkpoint_path, map_location=loc)
            #checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)#载入模型
            start_epoch = checkpoint['epoch']#载入epoch
            best_epoch = checkpoint['best_epoch']#载入best epoch
            best_acc1 = checkpoint['best_acc1']#载入top1 best_acc
            #print(f"模型导入完成，说明可以用，start_epoch:{start_epoch}")
            if gpu_id is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(gpu_id)
            optimizer.load_state_dict(checkpoint['optimizer'])#载入optimizer
            scheduler.load_state_dict(checkpoint['scheduler'])#载入scheduler
        else:
            logging.info(f"{gpu_info}=> no checkpoint found at '{save_path}'")
    
    logging.info(f"{gpu_info}=> Begin training.")
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, gpu_id, print_freq=cfg["train_print_freq"], gpu_info=gpu_info)
        scheduler.step()
        # acc1 = validate(val_loader, model, criterion, args)
        acc1, acc5 = validate(val_loader, model, criterion, gpu_id, print_freq=cfg["val_print_freq"], gpu_info=gpu_info)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            # record the best epoch
            best_epoch = epoch
        logging.info(f'{gpu_info}=>Best acc at epoch {best_epoch}: {best_acc1}')
        
    
        if not distributed:
            logger.log_metrics(epoch, acc1.item(), acc5.item())
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': quant_arch if quant else arch,
                'state_dict': model.state_dict(),
                'best_epoch': best_epoch,
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, save_path)
        elif distributed and rank == 0:
            logger.log_metrics(epoch, acc1.item(), acc5.item())
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': quant_arch if quant else arch,
                'state_dict': model.module.state_dict(),
                'best_epoch': best_epoch,
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, save_path)
    if distributed:
        destroy_process_group()
if __name__ == "__main__":
    cfg = yaml_load("./default.yaml")
    seed_all(seed=cfg['seed'])
    gpu = cfg["gpu"]
    if isinstance(gpu, str):
        print(f"=> Run by using DDP at gpu {gpu}.")
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        gpu_ids = list(map(int, gpu.split(",")))
        world_size = len(gpu_ids)
        mp.spawn(main, args=(world_size, gpu_ids, cfg), nprocs=world_size)
    else:
        main(None, -1, gpu, cfg)