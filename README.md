
## TF_EnhanceDPED project

- tensorflow implement of image enhancement base on dped .

### dirctory introduction

- models文件里存储训练保存的模型，两个子文件分别保存训练和验证过程的event文件，可以通过命令tensorboard --lodir "models/"对训练和验证的参数进行可视化;
- results文件保存验证过程中生成图像和目标图像的对比图;
- net,data,loss分别为网络，数据导入和损失定义的文件夹。vgg_pretrained放置训练好的vgg模型，metrics放置输出评价指标函数（如psnr,ssim);

### congfig

- 配置文件在experiments里面的config文件里;

### tain

- 点击tool文件下的train文件即可训练;

### up to optimization

- ps:训练是将一部分数据读入内存进行训练，每训练一定次数可以重新导入一部分数据加载到内存里;
