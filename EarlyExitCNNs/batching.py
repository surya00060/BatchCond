import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import argparse

import numpy as np
from model import *

device = 'cuda'
print('==> Preparing data..')
transform_train = transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomCrop(224, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
transform_test = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_train_32x32 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test_32x32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=4)
trainset_32x32 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train_32x32)
testset_32x32 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test_32x32)
ee_loader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building and Loading Early Exit Model..')
blocks = [3,4,6,3]
net = ResNet(BasicBlock, blocks, num_classes=10)
net = net.to('cuda')
checkpoint = torch.load('./checkpoint/best_ckpt_224.pth')
net.load_state_dict(checkpoint['net'])
net.ee_entropy_threshold = 0.3
print('==> Finished loading model..')
criterion = nn.CrossEntropyLoss()
## Predictor
print('==> Building Predictor..')
predictor = ResNetEntropyPredictor(BasicBlock, [1,1,4], sum(blocks))
predictor.to(device)
checkpoint=torch.load(f'./checkpoint/predictor_ckpt.pth')
predictor.load_state_dict(checkpoint['net'])
print('==> Finished loading predictor..')

    
@torch.inference_mode()
def measure_time_with_batch_size_1_inference():
    net.eval()
    net.half()
    net.ee_entropy_threshold = 0.3
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    ## Batch Size 1 Inference with No Ineffectual Computations
    ## Warmup Time
    batch_loader = torch.utils.data.DataLoader(testset, 1, shuffle=False, num_workers=4)
    for batch_idx, (inputs, targets) in enumerate(batch_loader):
        inputs, targets = inputs.to(device).half(), targets.to(device)
        logits, exits = net.ee_inference(inputs)
        if batch_idx == 10:
            break
    batch_times = []
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(batch_loader):
        inputs, targets = inputs.to(device).half(), targets.to(device)
        start_time.record()
        logits, exits = net.ee_inference(inputs)
        end_time.record()
        torch.cuda.synchronize()
        batch_times.append(start_time.elapsed_time(end_time))
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
    b1_inference = sum(batch_times)
    print(f"Batch Size: {1}  Batch Size 1 Inference Time: {b1_inference} ms")

def measure_time(batch_size):
    net.eval()
    net.half()
    net.ee_entropy_threshold = 0.3
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    ## Batched Inference with Random Batching + Compute Padding
    batch_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=4)
    for batch_idx, (inputs, targets) in enumerate(batch_loader):
        inputs, targets = inputs.to(device).half(), targets.to(device)
        logits, exits = net.early_exit_padding_inference(inputs)
        if batch_idx == 10:
            break
    batch_times = []
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(batch_loader):
        inputs, targets = inputs.to(device).half(), targets.to(device)
        start_time.record()
        logits, exits = net.early_exit_padding_inference(inputs)
        end_time.record()
        torch.cuda.synchronize()
        batch_times.append(start_time.elapsed_time(end_time))
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
    b_inference = sum(batch_times)
    avg_exits = sum([exit.sum().item() for exit in exits]) / len(exits)
    print(f"Batch Size: {batch_size} Random Batching with Compute Padding Time: {b_inference} ms")

def measure_time_with_simbatch_abr(batch_size): 
    net.eval()
    net.half()
    net.ee_entropy_threshold = 0.3
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    ## Batched Inference with SimBatch + ABR
    predictor.eval()
    predictor.half()
    ## SimBatch Runtime
    pred_batch_size = 10000
    exit_predictions = torch.zeros(len(testset))
    pred_times = []
    prediction_32x32_dataloader = torch.utils.data.DataLoader(testset_32x32, pred_batch_size, shuffle=False, num_workers=4)
    for batch_idx, (inputs, targets) in enumerate(prediction_32x32_dataloader):
        inputs, targets = inputs.to(device).half(), targets.to(device)
        entropy_preds = predictor(inputs)
        start_time.record()
        predictor(inputs)
        end_time.record()
        torch.cuda.synchronize()
        pred_times.append(start_time.elapsed_time(end_time))
        check_entropy_less_than_threshold = (entropy_preds < net.ee_entropy_threshold).int()
        exit_predictions[batch_idx*pred_batch_size:batch_idx*pred_batch_size+pred_batch_size] = (torch.argmax(check_entropy_less_than_threshold, dim=1) + 1).to("cpu")
    pred_time = sum(pred_times)
    sorted_indices = np.argsort(exit_predictions)
    exit_predictions = exit_predictions[sorted_indices]
    ordered_dataset = torch.utils.data.Subset(testset, sorted_indices)
    batch_loader = torch.utils.data.DataLoader(ordered_dataset, batch_size, shuffle=False, num_workers=4)
    batch_times = []
    for batch_idx, (inputs, targets) in enumerate(batch_loader):
        inputs, targets = inputs.to(device).half(), targets.to(device)
        logits, exits = net.early_exit_abr_inference(inputs)
        start_time.record()
        logits, exits = net.early_exit_abr_inference(inputs)
        end_time.record()
        torch.cuda.synchronize()
        batch_times.append(start_time.elapsed_time(end_time))
    batch_time = sum(batch_times)
    print(f"Batch Size: {batch_size} SimBatch + ABR: {pred_time + batch_time} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "BatchCond: Efficient Batched Inference in Conditional Neural Networks")
    parser.add_argument("--measure_batch_size_1_inference", action = "store_true", help = "Perform batched inference.")
    parser.add_argument("--measure_batched_inference", action = "store_true", help = "Perform batched inference.")
    parser.add_argument("--measure_simbatch_abr", action = "store_true", help = "Perform batched inference.")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size for batched inference.")

    args = parser.parse_args()
    if args.measure_batch_size_1_inference:
        measure_time_with_batch_size_1_inference()
    if args.measure_batched_inference:
        measure_time(args.batch_size)
    if args.measure_simbatch_abr:
        measure_time_with_simbatch_abr(args.batch_size)
    