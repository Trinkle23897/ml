# Paper reading

[TOC]

## CNN 分类

### LeNet

得分解释：得分高的神经元组合成的路径为一条可能的解释路径

bias/norm能够保证平移不变性

subsampling：减轻小变化所带来的影响，减小敏感性

### AlexNet

在cpu上做出图像变换，喂给gpu；测试的时候将图片经过不同变换之后取预测概率平均值

dropout：每个神经元dropout概率为0.5，测试的时候每个的输出×2

LRN不work

### ZFNet

反卷积：“Adaptive deconvolutional networks for mid and high level feature learning”，根据可视化的特征进行网络结构的修改

前面的层收敛，然后后面的层再收敛：越深收敛越慢，不是同时以相同速度收敛的（梯度弥散 => sudden jumps）

卷积核太大：会把noise（低频与高频）吸收进来，冲淡原始signal

？网络对图片操作具有鲁棒性（较高层次提取的feature对于平移和尺度变化具有不变性，对非中心对称的旋转不具有不变性）：是原始图像变换之后的数据喂的好？（AlexNet里面说没做图像变换效果很差） => 给出差异，神经网络能够从中学出共性

### NIN

对feature map进行1*1 conv，增加非线性性

### GoogLeNet v1

1*1 conv：降维，增加表达能力（本质是对feature map进行线性组合），称之为bottleneck

Inception modules 用在 higher layers，低层还是conv，说是technical reasons

提了一句在数据中随机增加光照，用来增加光照和颜色的不变性。[链接](http://blog.csdn.net/sheng_ai/article/details/40652193)

训练的时候在中间层“注入梯度”，乘上一个weight decay因子，加速收敛（在v3中指出这种做法只会在后期训练的时候提升网络性能，并且最底层的梯度注入是可以去掉的，只是起到正则化作用）

### VGG

Train：pre-init，先把小的训好，然后不断加入新的层/改变图像的缩放比例再训。但是提到了Xavier初始化(2010)发现他们可以不用这么干2333

？Table6：两个model的ensemble比七个还要好？

### GoogLeNet v2 (BN)

将输入数据归一化处理之后送进去训练

在每个mini-batch里面计算当前batch内部数据的均值和方差，然后对输入数据进行归一化（归一化是基于channel的，每个channel内部进行归一化），同时为了保持网络特征的参数的表达性和可视化意义就会随之被破坏了，因此需要对归一化后的输出x再学习一个scale A和shift B，y=Ax+B，然后将y经过激活函数以后输出到下一层

We adopt batch normalization (BN) right after each convolution and before activation.

### GoogLeNet v3

factorized convolution：n\*n conv可以拆成n\*1+1\*n合起来，这样参数从原来变为（对于低层不适用，中等大小feature map（12~20）上使用7\*1+1\*7效果最好）

辅助分类器起到的并不是梯度注入作用，实际上起到的是regularization的作用，因为在辅助分类器前添加dropout或者batch normalization后效果更佳。

模块换位置能够减少计算量

提出了Label Smoothing Regularization：不再是01，而是$\epsilon$的概率给其他label，防止模型把预测值过度集中在概率较大类别上，把一些概率分到其他概率较小类别上。

高分辨率输入对结果影响不是很大，因此R-CNN中可以考虑使用专门的high-cost的低分辨率输入网络，来对小目标物体进行检测

### ResNet

在输出层输出的不是当前网络学到的映射函数F(x)，而是x+F(x)，如果F(x)和x的channel不统一的话就用1*1的卷积核来做channel的对齐

层数越多，单层修改的信号越少	

### GoogLeNet v4

v3+ResNet

### SqueezeNet

5\*5 conv改成3\*3会增大计算量：25\*C\*N\*N<9\*C\*(1+C')\*N\*N，但是瓶颈不在于计算，在于读取数据

dense-sparse-dense：使用裁剪之后的模型为初始值，再次进行训练调优所有参数，正确率能够提升4.3%。 稀疏相当于一种正则化，有机会把解从局部极小中解放出来。

### MobileNet v1

将传统卷积操作分解：$D_K · D_K · M · N · D_F · D_F$，input channels $M$ , output channels $N$ , the kernel size $D_k × D_k$ and the feature map size $D_F × D_F$ ，变为(1) Depthwise Convolutional Filters: $D_K*D_K*1*M$; (2) Pointwise Convolution: $1\times 1\times M\times N$。把经典的卷积核在空域和channel维度上进行解耦，depthwise卷积在每个channel上单独做，做完以后再用1*1的卷积核做pointwise卷积，这样能很大程度上减少卷积需要的参数量：$D_K · D_K · M · D_F · D_F + M · N · D_F · D_F$，优化比例为$\frac{1}{D_K^2}+\frac{1}{C_{out}}\sim\frac{1}{D_K^2}$

可以在M和N之前乘上一个瘦身因子$\alpha$，不过性能可能会下降

与factorized convolution的具体比较：假设N=M，$D_K*1*N*N*2*D_F*D_F$，优化比例为$\frac{2}{D_K}$

### MobileNet v2

改进：变成纺锤状，先升维，DC，再降维

channel数目少的时候，受到ReLU的影响更大，因此对于网络中用于channel降维的层后面都不再使用ReLU处理，而是使其保持线性



## Detection

### R-CNN

### OverFeat

**将FC看成1\*1 conv**，能够处理任意图像大小输入，输出不止一张feature map；然后综合输出的feature map得到得分 => 避免了在原图上的裁剪，实际上是增加采样数，但不增加计算量

可以在pooling的feature map上面进行滑窗

### SPPNet

把出来的feature map统一scale成4\*4、2\*2和1\*1的，在上面做检测，使得任意大小的特征图都能够转换成固定大小的特征向量；检测完成之后再反推回去

理论上conv大小是下取整出来的，实际上是算完写死的

### Fast R-CNN

roi pool：将每个box均匀分成N×M块，每块进行max pooling，然后再将这些feature map送进去处理

### Faster R-CNN

把selective search的那些框定死，固定大小的size和scale ratio，反正后面会fine tune

### R-FCN

### FPN

### Mask R-CNN

### Focal Loss



## Camera

### DeepISP



### DBL

双边滤波器：平滑的地方高斯滤波，不平滑的不要动

It is often simpler to predict the transformation from input to output rather than predicting the output directly.

Guidance map有助于提升效果，前提是要符合论文中的假设

### Demosaicking and Denoising



### HDR+

使用Raw，很多曝光的trick，不用过曝的帧

在连拍照片中取前三张中sharpest的一张当做reference frame，之后把其他的frame合并至ref.，建立了从low level alignment field到high level，支持相邻frame像素偏移169pixel的合并

分场景建模（暗光，中等，较亮），曝光时间依次下降

后处理使用2D/3D FFT，看不太懂，最后基本按照原来的框架比如“black-level subtraction，white balance，demosaic，chroma denoise，color correction，local tone map，dehaze，global tone map，sharpen，hue-specific color adjustments，dithering，saturation”，好多专业术语……不是很懂

basic idea：合并frame可以用cnn并行合并，输入两张图输出一张；之后把合并完的frame送到FFT kernel进行后处理，最后fine tune

loss func改成NIMA

### Fast Burst Images Denoising

Fast Burst Images Denoising这篇文章是把相机运动和被拍摄物体运动分别计算，以此再将不同的frame进行对齐；根据空间上的相关性建立了Pyramid homography graph，针对图像结构的特性对其使用相应的策略进行滤波处理

之前的运动估计通常依靠光流（逐点运功估计）或者块匹配（常用于视频压缩的运动估计）的方式。然而它们的计算非常耗时，并且结果容易受到噪声的影响。袁路所在的研究组对相机运动和被拍摄物体运动进行了拆分计算。首先，他们将相机的运动参数化，建立出一个参数模型，从而能非常快的估计出具体运动的相关参数。而针对场景中的物体运动，他们则对每一帧的具体运动进行了分析，针对运动的快慢，算法都能相应地估计出运动物体的精确轮廓。在每个像素时间轴上相加平均的时候，那些位于运动物体轮廓区域的像素都会被排除掉。通过这种简单的方式，他们能有效的解决图片每个像素对齐的问题。除此之外，他们进一步考虑空间上信息相关性，建立了一个多尺度“金字塔”结构，根据图像结构（例如，平滑区域和边缘）的分析，自适应的选择相应的平滑策略，从而达到进一步去除残留噪声的目的。最终，他们的算法将时间轴和空间轴结合在一起，在校正了每一帧的运动之后，得到的便是一幅清晰的图像。



## RL

因果分析的方法框架主要有：

1. 反事实理论（Counterfactual Theory）
2. 非参数结构方程（Nonparametric Structural Equations）
3. 有向无环图（Directed Acyclic Graph)
4. 混合模型