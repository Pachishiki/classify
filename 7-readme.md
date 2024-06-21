## 接口类说明

接口类文件classify.py，实现了接口类ViolenceClass。

ViolenceClass的初始化函数。

```python
def __init__(self):
    num_classes=2 #分类的类型数
    # 确保模型在GPU上运行，如果没有GPU则在CPU上
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.ckpt_root = "./" #模型检查点位置，注意！！！这里要填写classfy.py文件所在文件夹的位置
    self.ckpt_path = self.ckpt_root + "train_logs/resnet18_pretrain_test/version_1/checkpoints/resnet18_pretrain_test-epoch=39-val_loss=0.03.ckpt" #最终检查点的模型，也就是最终训练好的最佳模型


    # 加载训练好的resnet18模型
    self.model = ViolenceClassifier.load_from_checkpoint(self.ckpt_path)
    
    # 将模型设置为评估模式
    self.model.eval()
    
    # 确保模型的所有层都不再更新
    for param in self.model.parameters():
        param.requires_grad = False
    
    # 将模型转移到指定设备
    self.model.to(self.device)
```



接口类ViolenceClass提供一个接口函数ViolenceClass.classify(self,imgs : torch.Tensor)，用于将Tensor化的图片导入到函数中，并且用已经训练好的模型来分辨，结果会输出对应的预测类别（整数0或1）。函数接收参数第一个为self，为ViolenceClass类，第二个为imgs，属于torch.Tensor类。最后输出preds，为列表类。

```python
def classify(self,imgs : torch.Tensor) -> list: 
    # 确保输入张量在正确的设备上
    imgs = imgs.to(self.device)
    
    # 如果需要，添加batch维度
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    
    # 通过模型进行预测
    with torch.no_grad():  # 不计算梯度以节省内存和计算资源
        outputs = self.model(imgs)
    
    # 假设模型的最后一层是输出logits的线性层
    # 我们需要应用softmax来获取概率分布
    probs = torch.nn.functional.softmax(outputs, dim=1)
    
    # 选择概率最大的类别作为预测结果
    preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
    
    return preds
```

函数调用示例：

```
a = ViolenceClass()
print(a.classify(tensor_image))
# 输出[1,1]
```