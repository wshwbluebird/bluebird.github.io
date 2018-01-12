---
layout:     post
title:      "MAtConvNet 引入"
subtitle:   "机器学习框架"
date:       2018-01-08
author:     "bluebird"
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - MatConvNet
    - ML FrameWork
---

> “新手上路. ”

## MatConvNet 介绍

Convolutional Neural Networks (CNN)已经在对计算机视觉邻域产生了非常深远的影响，而MatConvNet就是用MatLab工具实现的CNN框架。Matlab是当前计算机视觉领域非常主流的研究和开发工具，而集成在Matlab中的MatConvNet方便了科研人员对CNN的使用。

由于CNN往往需要很大的数据来实现，所以CNN库的实现需要保证他的计算效率，MatConvNet通过对自身的代码实现的不断优化，以及对GPU计算的支持成为了主流的CNN库。同时MatConvNet使用户通过使用matlab命令来调用构造块，如	convolution, normalisation 和 pooling等，这些块可以很容易地组合和扩展以创建CNN体系结构，matlab自身对GPU计算的支持，使得用户自己也可以用matlab语言写出计算效率很高的构造块。

MatConvNet是一个开源的框架，它可以从[官网](http://www.vlfeat.org/matconvnet)以及从Git Hub下载



## MatConvNet 安装

MatConvNet的安装非常简单，这里提供两种安装的方式，目前都是cpu的安装方式，关于gpu的安装范式可以参考[这里](http://www.vlfeat.org/matconvnet/install/)

##### mex安装

MEX从字面上是MATLAB和Executable 两个单词的缩写，MEX文件是一种可在matlab环境中调用的C语言（或fortran）衍生程序，mex的编译结果实际上就是一个带输出函数mexFunction 的dll文件。

简单的说，如果你用C或者C++实现了一个函数，然后想用matlab来调用，那么这个时候你可以通过把它转换成mex文件来达到这个目的，而mex命令可以把.c或者.cpp文件转换成mex文件。

mex要转换.c和.cpp文件需要一个能编译c和cpp文件的编译器，而这个编译器需要开发者在自己的环境中配置好。

1. 对于**macOS**系统，要保证 Xcode 是安装好的，这样mex可以通过clang来编译.c或者.cpp
2. 对于**Linux**系统，要确保 GCC 4.8 and LibJPEG 都已经安装好了。要安装LibJPEG，在Ubuntu/Debian类的操作系统上可以使 ```sudo apt-get install build-essential libjpeg-turbo8-dev ```命令，在Fedora/Centos/RedHat之类的操作系统，可以使用 ```sudo yum install gcc gcc-c++ libjpeg-turbo-devel``` 命令。老版本的GCC（比如4.7）可能在使用MatConvNet中出现不兼容的状况
3. 对于**Windows**系统，要确保安装了 Visual Studio 2015 或者更新的版本
4. 如果想要灵活的配置自己的编译器，可以到点[这里](http://cn.mathworks.com/help/matlab/matlab_external/changing-default-compiler.html)

在配置好编译器后，可以在matlab的命令行窗口执行下面的命令，来安装mex

```matlab
> mex -setup 
> mex -setup C++
```



##### 手动安装

1. 下载[源代码](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz),到自己指定的目录，假设解压后的文件命名为`<MatConvNet>`.

2. 编译源文件，在确保mex正确建立之后在Matlab命令行中使用如下命令进行源文件的编译

   ```matlab
   > cd <MatConvNet>
   > addpath matlab
   > vl_compilenn
   ```

   正常情况下，编译可以成功，如果编译出现了问题，可以通过``` vl_compilenn('verbose', 1)```来追踪出现问题的脚本

3. 安装编译好后的文件，在matlab命令行中执行如下命令（这一步也可以在代码中实现）

   ```matlab
   > run <MatConvNet>/matlab/vl_setupnn
   ```

4. 测试，可以通过使用在matlab命令行中执行如下命令进行测试

   ```matlab
   > vl_testnn
   ```

   ​

##### 自动安装

自动安装也是官方网站推荐的一种方式，并给出了一个使用网络模型进行物体识别的案例

1. 首先打开Matlab，进入到要安装MatConvNet的目标路径中，建立一个新的setup.m文件

2. setup.m文件的内容如下

   ```matlab
   % Install and compile MatConvNet (needed once).
   untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz') ;
   cd matconvnet-1.0-beta25
   run matlab/vl_compilenn ;

   % Download a pre-trained CNN from the web (needed once).
   urlwrite(...
     'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
     'imagenet-vgg-f.mat') ;

   % Setup MatConvNet.
   run matlab/vl_setupnn ;

   % Load a model and upgrade it to MatConvNet current version.
   net = load('imagenet-vgg-f.mat') ;
   net = vl_simplenn_tidy(net) ;

   % Obtain and preprocess an image.
   im = imread('peppers.png') ;
   im_ = single(im) ; % note: 255 range
   im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
   im_ = im_ - net.meta.normalization.averageImage ;

   % Run the CNN.
   res = vl_simplenn(net, im_) ;

   % Show the classification result.
   scores = squeeze(gather(res(end).x)) ;
   [bestScore, best] = max(scores) ;
   figure(1) ; clf ; imagesc(im) ;
   title(sprintf('%s (%d), score %.3f',...
      net.meta.classes.description{best}, best, bestScore)) ;
   ```

   这段代码的内容是首先下载MatConvNet的源代码，解压后进行编译（所以要确保之前mex已经配置好）之后下载一个提前训练好的CNN模型，然后安装编译好的MatConvNet的源代码。

   最后运行进行安装测试，测试跑的是一个识别物体的样例。上面的代码加载了一种那个图像，并用网络识别图像中的物体，打印识别的结果

3. 运行setup，等待结果，如果运行正常可以看到如下结果![结果](https://github.com/wshwbluebird/wshwbluebird.github.io/raw/master/matconvnet_img/setup%20result.png)

   ​		

### 引用

http://www.vlfeat.org/matconvnet/quick/

http://www.vlfeat.org/matconvnet/install/

http://www.robots.ox.ac.uk/~vedaldi/assets/teach/2015/vedaldi15aims-bigdata-lecture-4-deep-learning-handout.pdf

http://blog.sina.com.cn/s/blog_468651400100coas.html



​	