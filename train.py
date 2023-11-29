import os
import sys
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

    image_path = ("data/input")      # 训练图片存放路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    
    # 读取图片
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    train_num = len(train_dataset)    # 训练集图片个数
    val_num = len(validate_dataset)

    batch_size = 4             # 默认batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数
    print('Using {} dataloader workers every process'.format(nw))
    
    # 数据加载
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64, shuffle=True,          # 打乱训练数据
                                               num_workers=nw)         
    
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 网络搭建（调用ResNet50并进行权重初始化）
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    for param in net.parameters():       # 固定部分层的参数
        param.requires_grad = False

    # 修改全连接层
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    
    # 指定训练网络使用的设备
    net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 定义优化器（只训练部分层的参数）
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    epochs = 1     # 迭代次数
    best_acc = 0.0
    save_path = './checkpoints/resnet50_2023.11.17.pth'    # 训练模型保存路径
    train_steps = len(train_loader)  
    
    for epoch in range(epochs):
        
        # 训练网络
        net.train()      # 启用dropout
        running_loss = 0.0      # 统计训练过程中的平均损失
        train_bar = tqdm(train_loader, file=sys.stdout)   # 可视化网络训练进度
        
        # 遍历训练集
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()     # 梯度清零
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))      # 比较预测值与真实值，计算损失梯度
            loss.backward()           # 反向传导
            optimizer.step()          # 更新网络参数
            
            running_loss += loss.item()   # 损失累加
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
            
        # 交叉验证（在这个过程中不需要计算梯度）
        net.eval()       # 关闭dropout
        acc = 0.0        # 计算每个epoch预测正确的样本数
        with torch.no_grad():        
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))         
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

        val_accurate = acc / val_num      # 计算交叉验证集的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))    

         # 权重的更新
        if val_accurate > best_acc:        
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

if __name__ == '__main__':
    main()

