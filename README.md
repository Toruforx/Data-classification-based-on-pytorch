# 基于Pytorch的数据分类——CIFAR10

### **1. 研究目的**

本研究的目标是通过 PyTorch 实现一个卷积神经网络（ConvNet）模型，并使用该模型对 CIFAR-10 数据集进行图像分类。我们旨在探索卷积神经网络在小型图像数据集上的性能，并通过实验了解模型的训练过程和在测试集上的分类准确度。代码在https://github.com/Toruforx/Data-classification-based-on-pytorch。

### **2. 方法**

**数据预处理：** 我们使用了数据增广技术，通过填充、随机水平翻转和随机裁剪，对图像进行处理。这有助于扩充数据集，增加模型的泛化能力。

```python
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])
```

**模型结构：** 我们设计了一个包含两个卷积层和一个全连接层的卷积神经网络。

```python
class ConvNet(nn.Module):
    # 构造函数定义网络层
    # ...
```

**训练过程：** 在训练过程中，我们使用了 Adam 优化器和交叉熵损失函数。我们进行了若干个 epochs 的训练，并监控了每个 epoch 中的损失。

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # ...
```

### **3. 实验结果**

#### 3.1. 对于超参数num_epoches的修改

#### 第一组数据（num_epoches = 5）

在这组实验中，我们设置训练过程的epoch数量为5。以下是实验结果的总结：

- **训练过程**：
  - 训练损失从2.0671（第一个epoch）逐渐下降至1.0033（第五个epoch）。
  - 随着训练的进行，模型逐渐学习到数据的特征，导致损失的减小。
- **测试准确度**：
  - 模型在测试集上的准确度为64.54%。
  - 由于训练epoch较少，模型可能未能完全发掘数据的复杂特征，导致测试准确度相对较低。

#### 第二组数据（num_epoches = 10）

这次，我们将epoch数量增加到10，以观察模型性能的变化。

- **训练过程**：
  - 训练损失从2.1694（第一个epoch）下降至1.0299（第十个epoch）。
  - 相较于第一组数据，模型有更多机会学习数据的特征，损失减小更为显著。
- **测试准确度**：
  - 测试准确度提高到了一个相对较高的水平，为65.52%。
  - 随着epoch的增加，模型性能有所提升，但是否过拟合需要进一步考虑。

#### 第三组数据（num_epoches = 20）

在这组实验中，我们进一步增加epoch数量，看看是否能够继续提高模型性能。

- **训练过程**：
  - 训练损失从1.7665（第一个epoch）下降至0.9817（第二十个epoch）。
  - 损失的下降逐渐趋缓，表明模型已经逐渐收敛到一定程度。
- **测试准确度**：
  - 测试准确度达到了70.76%。
  - 尽管epoch数量增加，但测试准确度似乎没有进一步提高，可能出现了过拟合。

#### 总结

- 随着epoch数量的增加，训练损失逐渐减小，模型更好地学习到了数据的特征。
- 测试准确度在一定程度上随着epoch的增加而提高，但在一定阈值后可能趋于稳定。
- 需要权衡epoch数量，以避免过拟合。

#### 3.2. 对于超参数learning_rate的修改

#### 第一组数据（learning_rate = 0.001）

- **训练过程**：
  - 训练损失从2.0671（第一个epoch）逐渐下降至0.9753（第五个epoch）。
  - 学习率相对较小，模型在每个epoch内都有充分的时间减小损失。
- **测试准确度**：
  - 模型在测试集上的准确度为64.54%。
  - 较小的学习率可能导致模型收敛速度较慢，但结果表明模型在测试上表现尚可。

#### 第二组数据（learning_rate = 0.01）

- **训练过程**：
  - 训练损失从2.0285（第一个epoch）下降至0.8658（第五个epoch）。
  - 学习率增大，导致模型更快地学习到数据的特征，损失下降幅度相对更大。
- **测试准确度**：
  - 模型在测试集上的准确度为56.25%。
  - 学习率增大可能导致模型在一定程度上跳过最优点，出现性能下降。

#### 第三组数据（learning_rate = 0.1）

- **训练过程**：
  - 训练损失从2.3231（第一个epoch）下降至2.2856（第五个epoch）。
  - 学习率较大，导致模型在训练过程中波动较大，不容易收敛。
- **测试准确度**：
  - 模型在测试集上的准确度为10.0%。
  - 学习率过大可能导致模型无法收敛，性能极差。

#### 总结

- **学习率为0.001**时，模型性能较好，但可能需要更多的epoch。
- **学习率为0.01**时，模型训练速度较快，但测试准确度下降。
- **学习率为0.1**时，模型无法有效收敛，测试准确度非常低。

### **4. 结论和讨论**

**实验总结：** 通过实验，我们成功训练了一个卷积神经网络模型，用于对 CIFAR-10 数据集进行图像分类。通过调整模型的超参数，发现学习率为0.001，epoch为20时表现最好，达到了 70.76% 的准确度。

**改进和未来工作：** 为了进一步提高模型性能，我们建议尝试不同的超参数组合、更复杂的网络结构或其他优化技术。可以考虑使用更大的数据集或迁移学习等方法来改进模型的泛化能力。

**实验经验：** 我们发现数据预处理对模型性能的影响很大。适当的数据增广技术有助于提高模型的鲁棒性。

### 附录

```python
num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.001
Epoch [1/5], Step [100/1563], Loss: 2.0671
Epoch [1/5], Step [200/1563], Loss: 1.5331
Epoch [1/5], Step [300/1563], Loss: 1.8679
Epoch [1/5], Step [400/1563], Loss: 1.5865
Epoch [1/5], Step [500/1563], Loss: 1.7275
Epoch [1/5], Step [600/1563], Loss: 1.6228
Epoch [1/5], Step [700/1563], Loss: 1.5906
Epoch [1/5], Step [800/1563], Loss: 1.4937
Epoch [1/5], Step [900/1563], Loss: 1.2937
Epoch [1/5], Step [1000/1563], Loss: 1.4682
Epoch [1/5], Step [1100/1563], Loss: 1.2535
Epoch [1/5], Step [1200/1563], Loss: 1.7675
Epoch [1/5], Step [1300/1563], Loss: 1.5054
Epoch [1/5], Step [1400/1563], Loss: 1.3051
Epoch [1/5], Step [1500/1563], Loss: 1.2790
Epoch [2/5], Step [100/1563], Loss: 1.3750
Epoch [2/5], Step [200/1563], Loss: 1.4465
Epoch [2/5], Step [300/1563], Loss: 1.2707
Epoch [2/5], Step [400/1563], Loss: 1.1792
Epoch [2/5], Step [500/1563], Loss: 0.9998
Epoch [2/5], Step [600/1563], Loss: 1.3979
Epoch [2/5], Step [700/1563], Loss: 1.5285
Epoch [2/5], Step [800/1563], Loss: 1.1963
Epoch [2/5], Step [900/1563], Loss: 1.2829
Epoch [2/5], Step [1000/1563], Loss: 1.1316
Epoch [2/5], Step [1100/1563], Loss: 1.2769
Epoch [2/5], Step [1200/1563], Loss: 1.2222
Epoch [2/5], Step [1300/1563], Loss: 1.3740
Epoch [2/5], Step [1400/1563], Loss: 1.2187
Epoch [2/5], Step [1500/1563], Loss: 1.2821
Epoch [3/5], Step [100/1563], Loss: 1.1554
Epoch [3/5], Step [200/1563], Loss: 1.0911
Epoch [3/5], Step [300/1563], Loss: 1.4534
Epoch [3/5], Step [400/1563], Loss: 1.5170
Epoch [3/5], Step [500/1563], Loss: 1.0676
Epoch [3/5], Step [600/1563], Loss: 1.0082
Epoch [3/5], Step [700/1563], Loss: 1.1892
Epoch [3/5], Step [800/1563], Loss: 1.2957
Epoch [3/5], Step [900/1563], Loss: 1.4649
Epoch [3/5], Step [1000/1563], Loss: 1.0888
Epoch [3/5], Step [1100/1563], Loss: 1.3011
Epoch [3/5], Step [1200/1563], Loss: 1.1690
Epoch [3/5], Step [1300/1563], Loss: 1.0331
Epoch [3/5], Step [1400/1563], Loss: 0.9464
Epoch [3/5], Step [1500/1563], Loss: 0.8093
Epoch [4/5], Step [100/1563], Loss: 1.0293
Epoch [4/5], Step [200/1563], Loss: 0.9772
Epoch [4/5], Step [300/1563], Loss: 0.8738
Epoch [4/5], Step [400/1563], Loss: 0.8273
Epoch [4/5], Step [500/1563], Loss: 0.8214
Epoch [4/5], Step [600/1563], Loss: 1.2065
Epoch [4/5], Step [700/1563], Loss: 1.3179
Epoch [4/5], Step [800/1563], Loss: 0.9032
Epoch [4/5], Step [900/1563], Loss: 0.9904
Epoch [4/5], Step [1000/1563], Loss: 1.3627
Epoch [4/5], Step [1100/1563], Loss: 1.1602
Epoch [4/5], Step [1200/1563], Loss: 1.0685
Epoch [4/5], Step [1300/1563], Loss: 1.0316
Epoch [4/5], Step [1400/1563], Loss: 0.7979
Epoch [4/5], Step [1500/1563], Loss: 1.0388
Epoch [5/5], Step [100/1563], Loss: 0.9772
Epoch [5/5], Step [200/1563], Loss: 0.9687
Epoch [5/5], Step [300/1563], Loss: 0.6400
Epoch [5/5], Step [400/1563], Loss: 1.0681
Epoch [5/5], Step [500/1563], Loss: 0.7423
Epoch [5/5], Step [600/1563], Loss: 0.9565
Epoch [5/5], Step [700/1563], Loss: 0.9606
Epoch [5/5], Step [800/1563], Loss: 0.9792
Epoch [5/5], Step [900/1563], Loss: 0.9753
Epoch [5/5], Step [1000/1563], Loss: 1.6984
Epoch [5/5], Step [1100/1563], Loss: 1.1438
Epoch [5/5], Step [1200/1563], Loss: 0.7179
Epoch [5/5], Step [1300/1563], Loss: 0.9352
Epoch [5/5], Step [1400/1563], Loss: 1.3850
Epoch [5/5], Step [1500/1563], Loss: 1.0033
Test Accuracy of the model on the test images: 64.54 %
```

```python
num_epochs = 10
num_classes = 10
batch_size = 32
learning_rate = 0.001
Epoch [1/10], Step [100/1563], Loss: 2.1694
Epoch [1/10], Step [200/1563], Loss: 1.6951
Epoch [1/10], Step [300/1563], Loss: 1.6698
Epoch [1/10], Step [400/1563], Loss: 1.9191
Epoch [1/10], Step [500/1563], Loss: 1.0941
Epoch [1/10], Step [600/1563], Loss: 1.5306
Epoch [1/10], Step [700/1563], Loss: 1.3131
Epoch [1/10], Step [800/1563], Loss: 1.6067
Epoch [1/10], Step [900/1563], Loss: 1.6930
Epoch [1/10], Step [1000/1563], Loss: 1.9656
Epoch [1/10], Step [1100/1563], Loss: 1.1209
Epoch [1/10], Step [1200/1563], Loss: 1.3024
Epoch [1/10], Step [1300/1563], Loss: 1.0822
Epoch [1/10], Step [1400/1563], Loss: 1.5937
Epoch [1/10], Step [1500/1563], Loss: 1.2932
Epoch [2/10], Step [100/1563], Loss: 1.1298
Epoch [2/10], Step [200/1563], Loss: 1.3502
Epoch [2/10], Step [300/1563], Loss: 1.2095
Epoch [2/10], Step [400/1563], Loss: 1.1639
Epoch [2/10], Step [500/1563], Loss: 1.5729
Epoch [2/10], Step [600/1563], Loss: 1.4750
Epoch [2/10], Step [700/1563], Loss: 1.1593
Epoch [2/10], Step [800/1563], Loss: 0.8871
Epoch [2/10], Step [900/1563], Loss: 1.2536
Epoch [2/10], Step [1000/1563], Loss: 1.1451
Epoch [2/10], Step [1100/1563], Loss: 1.3340
Epoch [2/10], Step [1200/1563], Loss: 1.2218
Epoch [2/10], Step [1300/1563], Loss: 1.4294
Epoch [2/10], Step [1400/1563], Loss: 1.3969
Epoch [2/10], Step [1500/1563], Loss: 1.0920
Epoch [3/10], Step [100/1563], Loss: 1.2745
Epoch [3/10], Step [200/1563], Loss: 1.1423
Epoch [3/10], Step [300/1563], Loss: 1.2852
Epoch [3/10], Step [400/1563], Loss: 1.5664
Epoch [3/10], Step [500/1563], Loss: 1.2302
Epoch [3/10], Step [600/1563], Loss: 1.1868
Epoch [3/10], Step [700/1563], Loss: 1.1978
Epoch [3/10], Step [800/1563], Loss: 1.0045
Epoch [3/10], Step [900/1563], Loss: 1.1761
Epoch [3/10], Step [1000/1563], Loss: 1.0167
Epoch [3/10], Step [1100/1563], Loss: 1.0383
Epoch [3/10], Step [1200/1563], Loss: 1.1634
Epoch [3/10], Step [1300/1563], Loss: 0.8937
Epoch [3/10], Step [1400/1563], Loss: 0.9713
Epoch [3/10], Step [1500/1563], Loss: 1.0602
Epoch [4/10], Step [100/1563], Loss: 1.3883
Epoch [4/10], Step [200/1563], Loss: 1.3010
Epoch [4/10], Step [300/1563], Loss: 1.0900
Epoch [4/10], Step [400/1563], Loss: 1.1153
Epoch [4/10], Step [500/1563], Loss: 0.8182
Epoch [4/10], Step [600/1563], Loss: 1.0211
Epoch [4/10], Step [700/1563], Loss: 1.2634
Epoch [4/10], Step [800/1563], Loss: 1.0210
Epoch [4/10], Step [900/1563], Loss: 1.0536
Epoch [4/10], Step [1000/1563], Loss: 0.8628
Epoch [4/10], Step [1100/1563], Loss: 0.8974
Epoch [4/10], Step [1200/1563], Loss: 1.1680
Epoch [4/10], Step [1300/1563], Loss: 1.0015
Epoch [4/10], Step [1400/1563], Loss: 1.0093
Epoch [4/10], Step [1500/1563], Loss: 1.0921
Epoch [5/10], Step [100/1563], Loss: 1.2399
Epoch [5/10], Step [200/1563], Loss: 1.0492
Epoch [5/10], Step [300/1563], Loss: 0.9256
Epoch [5/10], Step [400/1563], Loss: 0.8103
Epoch [5/10], Step [500/1563], Loss: 1.4400
Epoch [5/10], Step [600/1563], Loss: 0.6578
Epoch [5/10], Step [700/1563], Loss: 1.4557
Epoch [5/10], Step [800/1563], Loss: 0.7744
Epoch [5/10], Step [900/1563], Loss: 0.9021
Epoch [5/10], Step [1000/1563], Loss: 1.1473
Epoch [5/10], Step [1100/1563], Loss: 1.2002
Epoch [5/10], Step [1200/1563], Loss: 0.7988
Epoch [5/10], Step [1300/1563], Loss: 1.3252
Epoch [5/10], Step [1400/1563], Loss: 1.1576
Epoch [5/10], Step [1500/1563], Loss: 1.0664
Epoch [6/10], Step [100/1563], Loss: 0.8232
Epoch [6/10], Step [200/1563], Loss: 1.0416
Epoch [6/10], Step [300/1563], Loss: 0.7675
Epoch [6/10], Step [400/1563], Loss: 1.0780
Epoch [6/10], Step [500/1563], Loss: 0.7554
Epoch [6/10], Step [600/1563], Loss: 0.8762
Epoch [6/10], Step [700/1563], Loss: 0.9693
Epoch [6/10], Step [800/1563], Loss: 1.1823
Epoch [6/10], Step [900/1563], Loss: 0.8710
Epoch [6/10], Step [1000/1563], Loss: 0.9666
Epoch [6/10], Step [1100/1563], Loss: 0.9125
Epoch [6/10], Step [1200/1563], Loss: 0.9052
Epoch [6/10], Step [1300/1563], Loss: 1.1700
Epoch [6/10], Step [1400/1563], Loss: 0.8750
Epoch [6/10], Step [1500/1563], Loss: 1.2722
Epoch [7/10], Step [100/1563], Loss: 0.8046
Epoch [7/10], Step [200/1563], Loss: 1.1817
Epoch [7/10], Step [300/1563], Loss: 0.9864
Epoch [7/10], Step [400/1563], Loss: 0.7776
Epoch [7/10], Step [500/1563], Loss: 0.8506
Epoch [7/10], Step [600/1563], Loss: 0.9119
Epoch [7/10], Step [700/1563], Loss: 0.7862
Epoch [7/10], Step [800/1563], Loss: 0.6369
Epoch [7/10], Step [900/1563], Loss: 0.9196
Epoch [7/10], Step [1000/1563], Loss: 0.9894
Epoch [7/10], Step [1100/1563], Loss: 0.9680
Epoch [7/10], Step [1200/1563], Loss: 1.1785
Epoch [7/10], Step [1300/1563], Loss: 0.8342
Epoch [7/10], Step [1400/1563], Loss: 0.9887
Epoch [7/10], Step [1500/1563], Loss: 0.7767
Epoch [8/10], Step [100/1563], Loss: 0.7009
Epoch [8/10], Step [200/1563], Loss: 1.0829
Epoch [8/10], Step [300/1563], Loss: 1.1872
Epoch [8/10], Step [400/1563], Loss: 0.7883
Epoch [8/10], Step [500/1563], Loss: 0.8345
Epoch [8/10], Step [600/1563], Loss: 0.6812
Epoch [8/10], Step [700/1563], Loss: 1.2791
Epoch [8/10], Step [800/1563], Loss: 1.0428
Epoch [8/10], Step [900/1563], Loss: 0.9358
Epoch [8/10], Step [1000/1563], Loss: 1.2766
Epoch [8/10], Step [1100/1563], Loss: 0.8355
Epoch [8/10], Step [1200/1563], Loss: 1.0654
Epoch [8/10], Step [1300/1563], Loss: 1.0757
Epoch [8/10], Step [1400/1563], Loss: 0.9539
Epoch [8/10], Step [1500/1563], Loss: 0.9752
Epoch [9/10], Step [100/1563], Loss: 0.8489
Epoch [9/10], Step [200/1563], Loss: 0.7669
Epoch [9/10], Step [300/1563], Loss: 1.3969
Epoch [9/10], Step [400/1563], Loss: 0.6937
Epoch [9/10], Step [500/1563], Loss: 1.2034
Epoch [9/10], Step [600/1563], Loss: 0.8159
Epoch [9/10], Step [700/1563], Loss: 1.2623
Epoch [9/10], Step [800/1563], Loss: 1.2783
Epoch [9/10], Step [900/1563], Loss: 0.8715
Epoch [9/10], Step [1000/1563], Loss: 1.1065
Epoch [9/10], Step [1100/1563], Loss: 0.9377
Epoch [9/10], Step [1200/1563], Loss: 0.8381
Epoch [9/10], Step [1300/1563], Loss: 0.9678
Epoch [9/10], Step [1400/1563], Loss: 0.9757
Epoch [9/10], Step [1500/1563], Loss: 0.7565
Epoch [10/10], Step [100/1563], Loss: 0.8890
Epoch [10/10], Step [200/1563], Loss: 1.1885
Epoch [10/10], Step [300/1563], Loss: 0.9676
Epoch [10/10], Step [400/1563], Loss: 0.9163
Epoch [10/10], Step [500/1563], Loss: 0.8307
Epoch [10/10], Step [600/1563], Loss: 0.6941
Epoch [10/10], Step [700/1563], Loss: 0.5827
Epoch [10/10], Step [800/1563], Loss: 0.7065
Epoch [10/10], Step [900/1563], Loss: 0.9797
Epoch [10/10], Step [1000/1563], Loss: 0.7437
Epoch [10/10], Step [1100/1563], Loss: 0.6285
Epoch [10/10], Step [1200/1563], Loss: 0.8040
Epoch [10/10], Step [1300/1563], Loss: 0.8484
Epoch [10/10], Step [1400/1563], Loss: 1.0741
Epoch [10/10], Step [1500/1563], Loss: 1.0299
Test Accuracy of the model on the test images: 65.52 %
```

```python
num_epochs = 20
num_classes = 10
batch_size = 32
learning_rate = 0.001
Epoch [1/20], Step [100/1563], Loss: 1.7665
Epoch [1/20], Step [200/1563], Loss: 1.7945
Epoch [1/20], Step [300/1563], Loss: 1.7122
Epoch [1/20], Step [400/1563], Loss: 1.5688
Epoch [1/20], Step [500/1563], Loss: 1.4478
Epoch [1/20], Step [600/1563], Loss: 1.4388
Epoch [1/20], Step [700/1563], Loss: 1.2051
Epoch [1/20], Step [800/1563], Loss: 1.4903
Epoch [1/20], Step [900/1563], Loss: 1.3379
Epoch [1/20], Step [1000/1563], Loss: 1.0758
Epoch [1/20], Step [1100/1563], Loss: 0.9692
Epoch [1/20], Step [1200/1563], Loss: 1.5096
Epoch [1/20], Step [1300/1563], Loss: 1.3601
Epoch [1/20], Step [1400/1563], Loss: 1.0077
Epoch [1/20], Step [1500/1563], Loss: 1.2882
Epoch [2/20], Step [100/1563], Loss: 1.3369
Epoch [2/20], Step [200/1563], Loss: 0.9702
Epoch [2/20], Step [300/1563], Loss: 1.2846
Epoch [2/20], Step [400/1563], Loss: 1.3124
Epoch [2/20], Step [500/1563], Loss: 1.1447
Epoch [2/20], Step [600/1563], Loss: 0.8160
Epoch [2/20], Step [700/1563], Loss: 1.2227
Epoch [2/20], Step [800/1563], Loss: 1.7262
Epoch [2/20], Step [900/1563], Loss: 0.9068
Epoch [2/20], Step [1000/1563], Loss: 1.3401
Epoch [2/20], Step [1100/1563], Loss: 1.3590
Epoch [2/20], Step [1200/1563], Loss: 1.3215
Epoch [2/20], Step [1300/1563], Loss: 1.2436
Epoch [2/20], Step [1400/1563], Loss: 0.9039
Epoch [2/20], Step [1500/1563], Loss: 0.9794
Epoch [3/20], Step [100/1563], Loss: 1.0246
Epoch [3/20], Step [200/1563], Loss: 1.2166
Epoch [3/20], Step [300/1563], Loss: 0.9232
Epoch [3/20], Step [400/1563], Loss: 0.8888
Epoch [3/20], Step [500/1563], Loss: 1.2065
Epoch [3/20], Step [600/1563], Loss: 1.1185
Epoch [3/20], Step [700/1563], Loss: 1.2304
Epoch [3/20], Step [800/1563], Loss: 1.3729
Epoch [3/20], Step [900/1563], Loss: 1.1593
Epoch [3/20], Step [1000/1563], Loss: 0.8180
Epoch [3/20], Step [1100/1563], Loss: 1.0723
Epoch [3/20], Step [1200/1563], Loss: 0.8773
Epoch [3/20], Step [1300/1563], Loss: 1.0972
Epoch [3/20], Step [1400/1563], Loss: 0.9487
Epoch [3/20], Step [1500/1563], Loss: 0.9299
Epoch [4/20], Step [100/1563], Loss: 0.7960
Epoch [4/20], Step [200/1563], Loss: 1.1859
Epoch [4/20], Step [300/1563], Loss: 0.5983
Epoch [4/20], Step [400/1563], Loss: 1.2247
Epoch [4/20], Step [500/1563], Loss: 1.2836
Epoch [4/20], Step [600/1563], Loss: 1.0799
Epoch [4/20], Step [700/1563], Loss: 0.6704
Epoch [4/20], Step [800/1563], Loss: 0.8007
Epoch [4/20], Step [900/1563], Loss: 0.9566
Epoch [4/20], Step [1000/1563], Loss: 1.1617
Epoch [4/20], Step [1100/1563], Loss: 0.9894
Epoch [4/20], Step [1200/1563], Loss: 1.1165
Epoch [4/20], Step [1300/1563], Loss: 0.9224
Epoch [4/20], Step [1400/1563], Loss: 1.0419
Epoch [4/20], Step [1500/1563], Loss: 1.0390
Epoch [5/20], Step [100/1563], Loss: 1.0556
Epoch [5/20], Step [200/1563], Loss: 0.8582
Epoch [5/20], Step [300/1563], Loss: 0.8342
Epoch [5/20], Step [400/1563], Loss: 0.5922
Epoch [5/20], Step [500/1563], Loss: 1.1352
Epoch [5/20], Step [600/1563], Loss: 0.8164
Epoch [5/20], Step [700/1563], Loss: 0.9431
Epoch [5/20], Step [800/1563], Loss: 0.9347
Epoch [5/20], Step [900/1563], Loss: 1.0253
Epoch [5/20], Step [1000/1563], Loss: 0.9726
Epoch [5/20], Step [1100/1563], Loss: 1.1551
Epoch [5/20], Step [1200/1563], Loss: 0.9549
Epoch [5/20], Step [1300/1563], Loss: 0.9904
Epoch [5/20], Step [1400/1563], Loss: 1.1592
Epoch [5/20], Step [1500/1563], Loss: 0.8458
Epoch [6/20], Step [100/1563], Loss: 0.9940
Epoch [6/20], Step [200/1563], Loss: 0.8821
Epoch [6/20], Step [300/1563], Loss: 0.9646
Epoch [6/20], Step [400/1563], Loss: 1.0450
Epoch [6/20], Step [500/1563], Loss: 1.1448
Epoch [6/20], Step [600/1563], Loss: 1.1101
Epoch [6/20], Step [700/1563], Loss: 0.9690
Epoch [6/20], Step [800/1563], Loss: 0.7914
Epoch [6/20], Step [900/1563], Loss: 0.8827
Epoch [6/20], Step [1000/1563], Loss: 0.9621
Epoch [6/20], Step [1100/1563], Loss: 0.9678
Epoch [6/20], Step [1200/1563], Loss: 1.1058
Epoch [6/20], Step [1300/1563], Loss: 1.0053
Epoch [6/20], Step [1400/1563], Loss: 0.8989
Epoch [6/20], Step [1500/1563], Loss: 0.8901
Epoch [7/20], Step [100/1563], Loss: 1.0322
Epoch [7/20], Step [200/1563], Loss: 0.9929
Epoch [7/20], Step [300/1563], Loss: 0.8568
Epoch [7/20], Step [400/1563], Loss: 0.9625
Epoch [7/20], Step [500/1563], Loss: 0.9744
Epoch [7/20], Step [600/1563], Loss: 0.7446
Epoch [7/20], Step [700/1563], Loss: 0.7838
Epoch [7/20], Step [800/1563], Loss: 1.4070
Epoch [7/20], Step [900/1563], Loss: 0.8335
Epoch [7/20], Step [1000/1563], Loss: 1.0983
Epoch [7/20], Step [1100/1563], Loss: 0.7729
Epoch [7/20], Step [1200/1563], Loss: 1.0679
Epoch [7/20], Step [1300/1563], Loss: 1.1984
Epoch [7/20], Step [1400/1563], Loss: 0.6939
Epoch [7/20], Step [1500/1563], Loss: 1.2930
Epoch [8/20], Step [100/1563], Loss: 1.2632
Epoch [8/20], Step [200/1563], Loss: 0.5793
Epoch [8/20], Step [300/1563], Loss: 1.5151
Epoch [8/20], Step [400/1563], Loss: 0.8981
Epoch [8/20], Step [500/1563], Loss: 1.1694
Epoch [8/20], Step [600/1563], Loss: 0.8374
Epoch [8/20], Step [700/1563], Loss: 0.6710
Epoch [8/20], Step [800/1563], Loss: 0.8504
Epoch [8/20], Step [900/1563], Loss: 0.7030
Epoch [8/20], Step [1000/1563], Loss: 1.1192
Epoch [8/20], Step [1100/1563], Loss: 0.9531
Epoch [8/20], Step [1200/1563], Loss: 1.0049
Epoch [8/20], Step [1300/1563], Loss: 0.6476
Epoch [8/20], Step [1400/1563], Loss: 0.5757
Epoch [8/20], Step [1500/1563], Loss: 1.1615
Epoch [9/20], Step [100/1563], Loss: 0.7888
Epoch [9/20], Step [200/1563], Loss: 0.7642
Epoch [9/20], Step [300/1563], Loss: 1.1392
Epoch [9/20], Step [400/1563], Loss: 0.5387
Epoch [9/20], Step [500/1563], Loss: 0.7076
Epoch [9/20], Step [600/1563], Loss: 1.0916
Epoch [9/20], Step [700/1563], Loss: 1.1233
Epoch [9/20], Step [800/1563], Loss: 0.8912
Epoch [9/20], Step [900/1563], Loss: 0.9545
Epoch [9/20], Step [1000/1563], Loss: 0.8221
Epoch [9/20], Step [1100/1563], Loss: 0.6319
Epoch [9/20], Step [1200/1563], Loss: 1.2830
Epoch [9/20], Step [1300/1563], Loss: 0.8004
Epoch [9/20], Step [1400/1563], Loss: 0.6422
Epoch [9/20], Step [1500/1563], Loss: 0.8504
Epoch [10/20], Step [100/1563], Loss: 1.2443
Epoch [10/20], Step [200/1563], Loss: 0.9796
Epoch [10/20], Step [300/1563], Loss: 0.8249
Epoch [10/20], Step [400/1563], Loss: 1.1530
Epoch [10/20], Step [500/1563], Loss: 0.7614
Epoch [10/20], Step [600/1563], Loss: 0.6412
Epoch [10/20], Step [700/1563], Loss: 0.9559
Epoch [10/20], Step [800/1563], Loss: 0.8888
Epoch [10/20], Step [900/1563], Loss: 0.8660
Epoch [10/20], Step [1000/1563], Loss: 0.7636
Epoch [10/20], Step [1100/1563], Loss: 1.2724
Epoch [10/20], Step [1200/1563], Loss: 1.0447
Epoch [10/20], Step [1300/1563], Loss: 0.8586
Epoch [10/20], Step [1400/1563], Loss: 0.8423
Epoch [10/20], Step [1500/1563], Loss: 0.8326
Epoch [11/20], Step [100/1563], Loss: 0.7086
Epoch [11/20], Step [200/1563], Loss: 0.6462
Epoch [11/20], Step [300/1563], Loss: 0.8763
Epoch [11/20], Step [400/1563], Loss: 0.8339
Epoch [11/20], Step [500/1563], Loss: 0.7531
Epoch [11/20], Step [600/1563], Loss: 1.0213
Epoch [11/20], Step [700/1563], Loss: 0.7797
Epoch [11/20], Step [800/1563], Loss: 1.0768
Epoch [11/20], Step [900/1563], Loss: 1.0148
Epoch [11/20], Step [1000/1563], Loss: 0.8590
Epoch [11/20], Step [1100/1563], Loss: 0.7671
Epoch [11/20], Step [1200/1563], Loss: 1.2466
Epoch [11/20], Step [1300/1563], Loss: 0.6911
Epoch [11/20], Step [1400/1563], Loss: 1.1124
Epoch [11/20], Step [1500/1563], Loss: 0.7608
Epoch [12/20], Step [100/1563], Loss: 0.9213
Epoch [12/20], Step [200/1563], Loss: 1.0664
Epoch [12/20], Step [300/1563], Loss: 1.3791
Epoch [12/20], Step [400/1563], Loss: 0.8332
Epoch [12/20], Step [500/1563], Loss: 0.9795
Epoch [12/20], Step [600/1563], Loss: 0.5355
Epoch [12/20], Step [700/1563], Loss: 0.5742
Epoch [12/20], Step [800/1563], Loss: 0.6323
Epoch [12/20], Step [900/1563], Loss: 1.1374
Epoch [12/20], Step [1000/1563], Loss: 0.7618
Epoch [12/20], Step [1100/1563], Loss: 1.1574
Epoch [12/20], Step [1200/1563], Loss: 0.7422
Epoch [12/20], Step [1300/1563], Loss: 0.7476
Epoch [12/20], Step [1400/1563], Loss: 0.8527
Epoch [12/20], Step [1500/1563], Loss: 0.6987
Epoch [13/20], Step [100/1563], Loss: 0.6742
Epoch [13/20], Step [200/1563], Loss: 0.8235
Epoch [13/20], Step [300/1563], Loss: 1.7316
Epoch [13/20], Step [400/1563], Loss: 0.9932
Epoch [13/20], Step [500/1563], Loss: 1.1032
Epoch [13/20], Step [600/1563], Loss: 1.1435
Epoch [13/20], Step [700/1563], Loss: 1.0167
Epoch [13/20], Step [800/1563], Loss: 0.6253
Epoch [13/20], Step [900/1563], Loss: 0.9201
Epoch [13/20], Step [1000/1563], Loss: 0.6595
Epoch [13/20], Step [1100/1563], Loss: 1.0511
Epoch [13/20], Step [1200/1563], Loss: 0.5457
Epoch [13/20], Step [1300/1563], Loss: 1.1425
Epoch [13/20], Step [1400/1563], Loss: 0.8620
Epoch [13/20], Step [1500/1563], Loss: 0.9903
Epoch [14/20], Step [100/1563], Loss: 0.9092
Epoch [14/20], Step [200/1563], Loss: 0.8214
Epoch [14/20], Step [300/1563], Loss: 1.2849
Epoch [14/20], Step [400/1563], Loss: 0.8323
Epoch [14/20], Step [500/1563], Loss: 0.7635
Epoch [14/20], Step [600/1563], Loss: 1.1819
Epoch [14/20], Step [700/1563], Loss: 1.0959
Epoch [14/20], Step [800/1563], Loss: 0.8289
Epoch [14/20], Step [900/1563], Loss: 0.5856
Epoch [14/20], Step [1000/1563], Loss: 0.9234
Epoch [14/20], Step [1100/1563], Loss: 0.8075
Epoch [14/20], Step [1200/1563], Loss: 0.7260
Epoch [14/20], Step [1300/1563], Loss: 0.8466
Epoch [14/20], Step [1400/1563], Loss: 0.7994
Epoch [14/20], Step [1500/1563], Loss: 0.8342
Epoch [15/20], Step [100/1563], Loss: 0.7220
Epoch [15/20], Step [200/1563], Loss: 0.8749
Epoch [15/20], Step [300/1563], Loss: 1.0812
Epoch [15/20], Step [400/1563], Loss: 0.8886
Epoch [15/20], Step [500/1563], Loss: 0.7961
Epoch [15/20], Step [600/1563], Loss: 0.8405
Epoch [15/20], Step [700/1563], Loss: 0.5428
Epoch [15/20], Step [800/1563], Loss: 0.8991
Epoch [15/20], Step [900/1563], Loss: 0.8403
Epoch [15/20], Step [1000/1563], Loss: 0.5706
Epoch [15/20], Step [1100/1563], Loss: 0.6346
Epoch [15/20], Step [1200/1563], Loss: 0.7680
Epoch [15/20], Step [1300/1563], Loss: 1.0464
Epoch [15/20], Step [1400/1563], Loss: 0.5847
Epoch [15/20], Step [1500/1563], Loss: 0.5762
Epoch [16/20], Step [100/1563], Loss: 0.8501
Epoch [16/20], Step [200/1563], Loss: 0.7375
Epoch [16/20], Step [300/1563], Loss: 0.7839
Epoch [16/20], Step [400/1563], Loss: 0.8398
Epoch [16/20], Step [500/1563], Loss: 0.6495
Epoch [16/20], Step [600/1563], Loss: 1.0695
Epoch [16/20], Step [700/1563], Loss: 0.5190
Epoch [16/20], Step [800/1563], Loss: 0.8318
Epoch [16/20], Step [900/1563], Loss: 0.8374
Epoch [16/20], Step [1000/1563], Loss: 0.8153
Epoch [16/20], Step [1100/1563], Loss: 0.9715
Epoch [16/20], Step [1200/1563], Loss: 0.7981
Epoch [16/20], Step [1300/1563], Loss: 0.8364
Epoch [16/20], Step [1400/1563], Loss: 0.6887
Epoch [16/20], Step [1500/1563], Loss: 0.8107
Epoch [17/20], Step [100/1563], Loss: 0.7895
Epoch [17/20], Step [200/1563], Loss: 0.7856
Epoch [17/20], Step [300/1563], Loss: 0.7993
Epoch [17/20], Step [400/1563], Loss: 0.7725
Epoch [17/20], Step [500/1563], Loss: 0.8679
Epoch [17/20], Step [600/1563], Loss: 1.1112
Epoch [17/20], Step [700/1563], Loss: 0.6621
Epoch [17/20], Step [800/1563], Loss: 0.6037
Epoch [17/20], Step [900/1563], Loss: 0.7914
Epoch [17/20], Step [1000/1563], Loss: 0.5516
Epoch [17/20], Step [1100/1563], Loss: 0.9770
Epoch [17/20], Step [1200/1563], Loss: 1.0518
Epoch [17/20], Step [1300/1563], Loss: 0.6884
Epoch [17/20], Step [1400/1563], Loss: 0.5042
Epoch [17/20], Step [1500/1563], Loss: 0.9839
Epoch [18/20], Step [100/1563], Loss: 0.6271
Epoch [18/20], Step [200/1563], Loss: 0.5573
Epoch [18/20], Step [300/1563], Loss: 0.8561
Epoch [18/20], Step [400/1563], Loss: 0.8301
Epoch [18/20], Step [500/1563], Loss: 0.9655
Epoch [18/20], Step [600/1563], Loss: 0.7015
Epoch [18/20], Step [700/1563], Loss: 0.8723
Epoch [18/20], Step [800/1563], Loss: 0.7721
Epoch [18/20], Step [900/1563], Loss: 0.8866
Epoch [18/20], Step [1000/1563], Loss: 0.7487
Epoch [18/20], Step [1100/1563], Loss: 1.0662
Epoch [18/20], Step [1200/1563], Loss: 0.8669
Epoch [18/20], Step [1300/1563], Loss: 0.6163
Epoch [18/20], Step [1400/1563], Loss: 0.6527
Epoch [18/20], Step [1500/1563], Loss: 0.7389
Epoch [19/20], Step [100/1563], Loss: 0.8425
Epoch [19/20], Step [200/1563], Loss: 0.6224
Epoch [19/20], Step [300/1563], Loss: 0.8820
Epoch [19/20], Step [400/1563], Loss: 0.5108
Epoch [19/20], Step [500/1563], Loss: 0.8584
Epoch [19/20], Step [600/1563], Loss: 1.1679
Epoch [19/20], Step [700/1563], Loss: 0.6887
Epoch [19/20], Step [800/1563], Loss: 0.8608
Epoch [19/20], Step [900/1563], Loss: 0.6696
Epoch [19/20], Step [1000/1563], Loss: 0.7986
Epoch [19/20], Step [1100/1563], Loss: 0.8823
Epoch [19/20], Step [1200/1563], Loss: 0.5841
Epoch [19/20], Step [1300/1563], Loss: 1.0363
Epoch [19/20], Step [1400/1563], Loss: 0.7548
Epoch [19/20], Step [1500/1563], Loss: 0.7474
Epoch [20/20], Step [100/1563], Loss: 0.6860
Epoch [20/20], Step [200/1563], Loss: 0.8508
Epoch [20/20], Step [300/1563], Loss: 0.7737
Epoch [20/20], Step [400/1563], Loss: 0.4875
Epoch [20/20], Step [500/1563], Loss: 0.7404
Epoch [20/20], Step [600/1563], Loss: 0.6811
Epoch [20/20], Step [700/1563], Loss: 0.9483
Epoch [20/20], Step [800/1563], Loss: 0.8856
Epoch [20/20], Step [900/1563], Loss: 0.7915
Epoch [20/20], Step [1000/1563], Loss: 0.4201
Epoch [20/20], Step [1100/1563], Loss: 0.6973
Epoch [20/20], Step [1200/1563], Loss: 0.4288
Epoch [20/20], Step [1300/1563], Loss: 0.9162
Epoch [20/20], Step [1400/1563], Loss: 0.8627
Epoch [20/20], Step [1500/1563], Loss: 0.9817
Test Accuracy of the model on the test images: 70.76 %
```

```python
num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.01
Epoch [1/5], Step [100/1563], Loss: 2.0285
Epoch [1/5], Step [200/1563], Loss: 1.7257
Epoch [1/5], Step [300/1563], Loss: 1.6597
Epoch [1/5], Step [400/1563], Loss: 1.7160
Epoch [1/5], Step [500/1563], Loss: 1.6929
Epoch [1/5], Step [600/1563], Loss: 1.5609
Epoch [1/5], Step [700/1563], Loss: 1.5520
Epoch [1/5], Step [800/1563], Loss: 1.4643
Epoch [1/5], Step [900/1563], Loss: 1.7709
Epoch [1/5], Step [1000/1563], Loss: 1.7826
Epoch [1/5], Step [1100/1563], Loss: 1.4832
Epoch [1/5], Step [1200/1563], Loss: 1.5795
Epoch [1/5], Step [1300/1563], Loss: 1.6144
Epoch [1/5], Step [1400/1563], Loss: 1.6278
Epoch [1/5], Step [1500/1563], Loss: 1.6032
Epoch [2/5], Step [100/1563], Loss: 1.3851
Epoch [2/5], Step [200/1563], Loss: 1.5289
Epoch [2/5], Step [300/1563], Loss: 1.2216
Epoch [2/5], Step [400/1563], Loss: 1.5833
Epoch [2/5], Step [500/1563], Loss: 1.2876
Epoch [2/5], Step [600/1563], Loss: 1.2230
Epoch [2/5], Step [700/1563], Loss: 1.2928
Epoch [2/5], Step [800/1563], Loss: 0.9035
Epoch [2/5], Step [900/1563], Loss: 1.0521
Epoch [2/5], Step [1000/1563], Loss: 1.2457
Epoch [2/5], Step [1100/1563], Loss: 0.8153
Epoch [2/5], Step [1200/1563], Loss: 1.7211
Epoch [2/5], Step [1300/1563], Loss: 1.3643
Epoch [2/5], Step [1400/1563], Loss: 1.3071
Epoch [2/5], Step [1500/1563], Loss: 1.2050
Epoch [3/5], Step [100/1563], Loss: 1.2575
Epoch [3/5], Step [200/1563], Loss: 1.2747
Epoch [3/5], Step [300/1563], Loss: 1.0428
Epoch [3/5], Step [400/1563], Loss: 1.3612
Epoch [3/5], Step [500/1563], Loss: 1.3064
Epoch [3/5], Step [600/1563], Loss: 0.9597
Epoch [3/5], Step [700/1563], Loss: 1.1361
Epoch [3/5], Step [800/1563], Loss: 1.5945
Epoch [3/5], Step [900/1563], Loss: 1.0688
Epoch [3/5], Step [1000/1563], Loss: 1.0036
Epoch [3/5], Step [1100/1563], Loss: 0.6902
Epoch [3/5], Step [1200/1563], Loss: 1.2935
Epoch [3/5], Step [1300/1563], Loss: 1.1179
Epoch [3/5], Step [1400/1563], Loss: 0.9368
Epoch [3/5], Step [1500/1563], Loss: 1.2663
Epoch [4/5], Step [100/1563], Loss: 1.1921
Epoch [4/5], Step [200/1563], Loss: 1.0145
Epoch [4/5], Step [300/1563], Loss: 1.3172
Epoch [4/5], Step [400/1563], Loss: 1.1199
Epoch [4/5], Step [500/1563], Loss: 1.4311
Epoch [4/5], Step [600/1563], Loss: 1.5106
Epoch [4/5], Step [700/1563], Loss: 0.8285
Epoch [4/5], Step [800/1563], Loss: 1.1877
Epoch [4/5], Step [900/1563], Loss: 0.9643
Epoch [4/5], Step [1000/1563], Loss: 0.9548
Epoch [4/5], Step [1100/1563], Loss: 0.8353
Epoch [4/5], Step [1200/1563], Loss: 1.1002
Epoch [4/5], Step [1300/1563], Loss: 1.3934
Epoch [4/5], Step [1400/1563], Loss: 1.1300
Epoch [4/5], Step [1500/1563], Loss: 1.2688
Epoch [5/5], Step [100/1563], Loss: 0.8674
Epoch [5/5], Step [200/1563], Loss: 1.2538
Epoch [5/5], Step [300/1563], Loss: 0.9660
Epoch [5/5], Step [400/1563], Loss: 1.5359
Epoch [5/5], Step [500/1563], Loss: 1.0738
Epoch [5/5], Step [600/1563], Loss: 0.9772
Epoch [5/5], Step [700/1563], Loss: 1.3751
Epoch [5/5], Step [800/1563], Loss: 1.1741
Epoch [5/5], Step [900/1563], Loss: 1.2153
Epoch [5/5], Step [1000/1563], Loss: 0.8658
Epoch [5/5], Step [1100/1563], Loss: 0.8789
Epoch [5/5], Step [1200/1563], Loss: 0.9664
Epoch [5/5], Step [1300/1563], Loss: 1.5347
Epoch [5/5], Step [1400/1563], Loss: 0.9856
Epoch [5/5], Step [1500/1563], Loss: 1.4043
Test Accuracy of the model on the test images: 56.25 %
```

```python
num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.1
Epoch [1/5], Step [100/1563], Loss: 2.3231
Epoch [1/5], Step [200/1563], Loss: 2.3597
Epoch [1/5], Step [300/1563], Loss: 2.2484
Epoch [1/5], Step [400/1563], Loss: 2.3014
Epoch [1/5], Step [500/1563], Loss: 2.3240
Epoch [1/5], Step [600/1563], Loss: 2.3164
Epoch [1/5], Step [700/1563], Loss: 2.3346
Epoch [1/5], Step [800/1563], Loss: 2.3082
Epoch [1/5], Step [900/1563], Loss: 2.3392
Epoch [1/5], Step [1000/1563], Loss: 2.3325
Epoch [1/5], Step [1100/1563], Loss: 2.3186
Epoch [1/5], Step [1200/1563], Loss: 2.2803
Epoch [1/5], Step [1300/1563], Loss: 2.3103
Epoch [1/5], Step [1400/1563], Loss: 2.3250
Epoch [1/5], Step [1500/1563], Loss: 2.2402
Epoch [2/5], Step [100/1563], Loss: 2.3419
Epoch [2/5], Step [200/1563], Loss: 2.3833
Epoch [2/5], Step [300/1563], Loss: 2.2676
Epoch [2/5], Step [400/1563], Loss: 2.3098
Epoch [2/5], Step [500/1563], Loss: 2.2759
Epoch [2/5], Step [600/1563], Loss: 2.3259
Epoch [2/5], Step [700/1563], Loss: 2.3128
Epoch [2/5], Step [800/1563], Loss: 2.3176
Epoch [2/5], Step [900/1563], Loss: 2.3050
Epoch [2/5], Step [1000/1563], Loss: 2.2860
Epoch [2/5], Step [1100/1563], Loss: 2.3458
Epoch [2/5], Step [1200/1563], Loss: 2.2993
Epoch [2/5], Step [1300/1563], Loss: 2.3212
Epoch [2/5], Step [1400/1563], Loss: 2.3473
Epoch [2/5], Step [1500/1563], Loss: 2.2666
Epoch [3/5], Step [100/1563], Loss: 2.2689
Epoch [3/5], Step [200/1563], Loss: 2.3417
Epoch [3/5], Step [300/1563], Loss: 2.3112
Epoch [3/5], Step [400/1563], Loss: 2.3032
Epoch [3/5], Step [500/1563], Loss: 2.3042
Epoch [3/5], Step [600/1563], Loss: 2.2762
Epoch [3/5], Step [700/1563], Loss: 2.3398
Epoch [3/5], Step [800/1563], Loss: 2.3096
Epoch [3/5], Step [900/1563], Loss: 2.2963
Epoch [3/5], Step [1000/1563], Loss: 2.3529
Epoch [3/5], Step [1100/1563], Loss: 2.3083
Epoch [3/5], Step [1200/1563], Loss: 2.2722
Epoch [3/5], Step [1300/1563], Loss: 2.3508
Epoch [3/5], Step [1400/1563], Loss: 2.3102
Epoch [3/5], Step [1500/1563], Loss: 2.3477
Epoch [4/5], Step [100/1563], Loss: 2.3055
Epoch [4/5], Step [200/1563], Loss: 2.3510
Epoch [4/5], Step [300/1563], Loss: 2.2676
Epoch [4/5], Step [400/1563], Loss: 2.3395
Epoch [4/5], Step [500/1563], Loss: 2.2794
Epoch [4/5], Step [600/1563], Loss: 2.3678
Epoch [4/5], Step [700/1563], Loss: 2.3404
Epoch [4/5], Step [800/1563], Loss: 2.3449
Epoch [4/5], Step [900/1563], Loss: 2.2548
Epoch [4/5], Step [1000/1563], Loss: 2.2966
Epoch [4/5], Step [1100/1563], Loss: 2.2806
Epoch [4/5], Step [1200/1563], Loss: 2.3018
Epoch [4/5], Step [1300/1563], Loss: 2.3549
Epoch [4/5], Step [1400/1563], Loss: 2.3194
Epoch [4/5], Step [1500/1563], Loss: 2.3266
Epoch [5/5], Step [100/1563], Loss: 2.3317
Epoch [5/5], Step [200/1563], Loss: 2.3222
Epoch [5/5], Step [300/1563], Loss: 2.3065
Epoch [5/5], Step [400/1563], Loss: 2.3300
Epoch [5/5], Step [500/1563], Loss: 2.3559
Epoch [5/5], Step [600/1563], Loss: 2.3580
Epoch [5/5], Step [700/1563], Loss: 2.3490
Epoch [5/5], Step [800/1563], Loss: 2.3045
Epoch [5/5], Step [900/1563], Loss: 2.3062
Epoch [5/5], Step [1000/1563], Loss: 2.3491
Epoch [5/5], Step [1100/1563], Loss: 2.3309
Epoch [5/5], Step [1200/1563], Loss: 2.3175
Epoch [5/5], Step [1300/1563], Loss: 2.2886
Epoch [5/5], Step [1400/1563], Loss: 2.2862
Epoch [5/5], Step [1500/1563], Loss: 2.2856
Test Accuracy of the model on the test images: 10.0 %
```