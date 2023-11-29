import os
import time
import torch
from PIL import Image
from torchvision import transforms
import torchvision

def main():
    start_time = time.time()  # 记录开始时间
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义数据预处理
    data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 测试集路径
    test_dir = './data/input/testset/newest_test'
    assert os.path.exists(test_dir), "directory: '{}' does not exist.".format(test_dir)
    image_path_all = os.listdir(test_dir)
    image_path_all.sort(key=lambda x:int(x.split('.')[-2].split('_')[-1]))

    # 初始化网络
    model = torchvision.models.resnet50(num_classes=2).to(device)

    # 加载模型权重
    weights_path = "./checkpoints/resnet50_2023.11.16.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()   # 关闭dropout

    # 测试并将结果以{file label}的形式写入score.txt文件
    output_dir = "./data/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "score.txt")

    with open(output_path, "w") as f:
        for file in image_path_all:
            img_path = os.path.join(test_dir, file)
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)
            output = torch.squeeze(model(img)).cpu()       # 降维得到最最终的输出
            predict = torch.softmax(output, dim=0)         # 将输出变为概率分布
            predict_cla = torch.argmax(predict).item()     # 获取概率最大处对应的值
            f.write("{} ".format(file))
            f.write("{}\n".format(predict_cla))
        f.write("Model Execution Time: {:.2f} seconds\n".format(time.time() - start_time))  # 将运行时间写入文件

if __name__ == '__main__':
  main()

