# Visualizing and Understanding

- 可视化第一层
- 使用最后一层来做knn
- 降维：最后一层 (t-SNE)
- 查看中间层对于输入图像的响应程度
- 遮挡图像看输出分类误差
- 反向计算图像梯度（Saliency Maps）：Segmentation without supervision
- Gradient Ascent：训练图像pixel将其得分最大化（其实不用pixel换成FC层效果更好）=> 对抗样本