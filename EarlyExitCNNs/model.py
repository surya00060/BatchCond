import torch
import torch.nn as nn
import torch.nn.functional as F

def entropy(x):
    x = torch.softmax(x, dim=-1)
    entropy = -torch.sum(x * torch.log(x), dim=-1)
    return entropy

# ResNet Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class EEBlock(nn.Module):
    def __init__(self,in_channels, num_classes):
        super(EEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self,x):
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()        
        self.inplanes = 64
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.stages = nn.ModuleList()
        self._make_layer(self.stages, block, [64, 128, 256, 512], num_blocks, [1,2,2,2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)

        self.ee_entropy_threshold = None
        self.decision_to_whether_drop = torch.load('decision_to_whether_drop.pt')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, module_list, block, planes_list, num_blocks_list, strides_list, num_classes=10):
        for i in range(len(num_blocks_list)):
            strides = [strides_list[i]] + [1]*(num_blocks_list[i]-1)
            for stride in strides:
                module_list.append(block(self.inplanes, planes_list[i], stride))
                module_list.append(EEBlock(planes_list[i], num_classes))
                self.inplanes = planes_list[i]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ee_logits = []
        for i in range(sum(self.num_blocks)):
            x = self.stages[2*i](x)
            ee_logits.append(self.stages[2*i+1](x))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, ee_logits
    
    @torch.inference_mode()
    def collect_entropy(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ee_entropies = torch.zeros((x.shape[0], sum(self.num_blocks)), device=x.device, dtype=x.dtype)
        for i in range(sum(self.num_blocks)):
            x = self.stages[2*i](x)
            ee_logits = self.stages[2*i+1](x)
            ee_entropies[:, i] = entropy(ee_logits)
        return ee_entropies

    @torch.inference_mode()
    def ee_inference(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(sum(self.num_blocks)):
            x = self.stages[2*i](x)
            ee_logits = self.stages[2*i+1](x)
            if entropy(ee_logits) < self.ee_entropy_threshold:
                return ee_logits, i+1
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, sum(self.num_blocks)+1
    
    @torch.inference_mode()
    def early_exit_padding_inference(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        sample_logits = torch.zeros((x.shape[0], self.num_classes), device=x.device, dtype=x.dtype)
        sample_exits = torch.zeros(x.shape[0], device=x.device, dtype=torch.int)
        already_exited = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        early_exit = False
        for i in range(sum(self.num_blocks)):
            x = self.stages[2*i](x)
            ee_logits = self.stages[2*i+1](x)
            ee_entropy = entropy(ee_logits)
            exit_mask = ee_entropy < self.ee_entropy_threshold
            samples_exiting = exit_mask & ~already_exited
            sample_exits[samples_exiting] = i+1
            sample_logits[samples_exiting] = ee_logits[samples_exiting]
            already_exited |= exit_mask
            if already_exited.all():
                early_exit = True
                break
        if not early_exit:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            sample_logits[~already_exited] = x[~already_exited]
            sample_exits[~already_exited] = sum(self.num_blocks)+1
        return sample_logits, sample_exits
    
    @torch.inference_mode()
    def early_exit_splitting_inference(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        sample_logits = torch.zeros((x.shape[0], self.num_classes), device=x.device, dtype=x.dtype)
        sample_exits = torch.zeros(x.shape[0], device=x.device, dtype=torch.int)
        sample_index = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        early_exit = False
        for i in range(sum(self.num_blocks)):
            x = self.stages[2*i](x)
            ee_logits = self.stages[2*i+1](x)
            ee_entropy = entropy(ee_logits)
            exit_mask = ee_entropy < self.ee_entropy_threshold
            if exit_mask.sum() == x.shape[0]:
                sample_logits[sample_index] = ee_logits
                sample_exits[sample_index] = i+1
                early_exit = True
                break
            if exit_mask.sum() > 0:
                x = x[~exit_mask]
                sample_logits[sample_index[exit_mask]] = ee_logits[exit_mask]
                sample_exits[sample_index[exit_mask]] = i+1
                sample_index = sample_index[~exit_mask]
        if not early_exit:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            sample_logits[sample_index] = x
            sample_exits[sample_index] = sum(self.num_blocks)+1
        return sample_logits, sample_exits
    
    @torch.inference_mode()
    def early_exit_abr_inference(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        sample_logits = torch.zeros((x.shape[0], self.num_classes), device=x.device, dtype=x.dtype)
        sample_exits = torch.zeros(x.shape[0], device=x.device, dtype=torch.int)
        already_exited = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        sample_index = torch.arange(x.shape[0], device=x.device, dtype=torch.int)

        early_exit = False
        for i in range(sum(self.num_blocks)):
            x = self.stages[2*i](x)
            ee_logits = self.stages[2*i+1](x)
            ee_entropy = entropy(ee_logits)
            exit_mask = ee_entropy < self.ee_entropy_threshold

            exit_this_layer = exit_mask & ~already_exited[sample_index]
            num_samples = x.shape[0]
            cond1 = exit_this_layer.sum() > 0
            if not cond1:
                continue
            selected_indices = sample_index[exit_this_layer]
            cond2 = num_samples > 16
            cond3 = self.decision_to_whether_drop[i][num_samples-1][exit_this_layer.sum()-1]
            if cond2 and cond3:
                x = x[~exit_this_layer]
                sample_logits[selected_indices] = ee_logits[exit_this_layer]
                sample_exits[selected_indices] = i+1
                already_exited[selected_indices] = True
                sample_index = sample_index[~exit_this_layer]
            else:
                sample_logits[selected_indices] = ee_logits[exit_this_layer]
                sample_exits[selected_indices] = i+1
                already_exited[selected_indices] = True
            
            if already_exited.all():
                early_exit = True
                break
        
        if not early_exit:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            exit_this_layer = ~already_exited[sample_index]
            sample_logits[sample_index[exit_this_layer]] = x[exit_this_layer]
            sample_exits[sample_index[exit_this_layer]] = sum(self.num_blocks)+1
        return sample_logits, sample_exits


class ResNetEntropyPredictor(nn.Module):
    def __init__(self, block, num_blocks, num_exits=16):
        super(ResNetEntropyPredictor, self).__init__()
        self.in_planes = 16
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layers = nn.ModuleList()
        self._make_layer(self.layers, block, [16, 32, 64], num_blocks, [1, 2, 2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regression = nn.Linear(64, num_exits)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, layers, block, planes_list, num_blocks_list, strides_list):
        for i in range(len(num_blocks_list)):
            strides = [strides_list[i]] + [1]*(num_blocks_list[i]-1)
            for stride in strides:
                layers.append(block(self.in_planes, planes_list[i], stride))
                self.in_planes = planes_list[i]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i, layer in enumerate(self.layers):
            out = layer(out)
        out = self.avgpool(out)
        return self.regression(out.view(out.size(0), -1))


