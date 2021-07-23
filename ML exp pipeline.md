目前方案： pytorch lightning Trainer + 第三方log工具 wandb(Weights and Biases)

## data

如果是新奇的实验领域，那么需要使用自己收集的数据集，一切都要重新定义。但是大多数情况下并非如此。

绝大部分实验都是使用现成的，有完成结构的数据集，在 pytorch 官方网站上可以看到很多。所以要做的就是熟悉调用的 API， 知道数据集的数据格式，数据包含什么内容(image, caption, video, text……), 知道如何对数据集进行 slicing, 知道如何从数据集中进行采样， 并且最后把数据集包装进 pytorch dataloader(甚至是 pl 的 DataModule). 

另外一个重要的方面是 data augmentation. 大部分情况下应该可以使用组合的 transform 完成，将定义好的 transform 传入 dataloader, 在加载数据的时候自动使用 data aug. 

如果需要对数据进行预处理，那么就把它们留在数据集加载的部分，在 dataloder sample data 之前。sample 出来的 data 应该已经是有良好结构的、准备直接送入 model 里面进行运算的了。不要让数据处理的代码掺杂到模型的 forward 方法或者其他地方。输入模型的 x 应该是有固定的 shape (比如 (B, C, H, W)) 的 Tensor. 

在开始整个训练之前，应该对数据集进行整体检视，让自己有一个感性上的认识。如果是图像分类的数据集，那么就 sample 一些 image 和他们的 label, 并且用表格的形式呈现出来；如果是 detection 或者 segmentation 的数据集，那么需要做的事情更多. 不过如果把过程抽象出来，那就是：sample, 然后 plot. 

用什么工具？通常第三方工具会比较好一些，自己知道的仅仅是 wandb 可以支持这样的数据可视化。使用 matplotlib 会花去自己大量的时间，因为需要熟悉制作合理的数据呈现方式的代码。**关于这一部分还需要不断前行，继续寻找更好的工具。** 

## model

model 的构建是整个 pipeline 最主要的部分，不过需要强调的地方其实不多，因为这也是平时最受重视的部分。但是，有如下几点：

1.   保持 model 结构的模块化，模块的结构要定义良好
2.   万万不要每次都从头开始构建 model. 在 github 上寻找一些优秀的模型源码并收藏或保存，或者自己编写一些经常用到的基本模块(比如 attention)，将他们保存下来，创建一个专门用于存放 template 的 github 仓库。甚至是打包成 python 的 package 上传到 pypi 上，这样就可以在命令行 pip install 并且在虚拟环境中使用，而不是每次都复制之前的代码。
3.   添加足够的注释…… 这已经是一个老生常谈的问题，不过对于我自己来说，有很大一部分的时间都花费在推导数据在模型中流动时的维度变化上。(导致数据维度变化的操作很多(几乎是全部)，而我们的数据通常又有很多个维度……) 因此，我自己喜欢用注释注明数据在操作之后的维度变化(特别是那些复杂的、看起来不那么明显的操作，这时候除了注明数据的维度最好再解释一下这一块对应模型中的什么部分)。
4.   type hint. 在函数的接口部分写上 type hint 能够省下重新推导参数类型的时间。 甚至, type hint 可以让 IDE 知道数据是什么类型，从而给出合理的代码提示。(标注为 Tensor, 就会收到 Tensor method 的提示， 而不是 Any 的什么都没有)
5.   DocString. 同样，是节省时间的方式…… 乍一看，这可能是浪费时间的工作，但是一个 model 通常都包含了很多部分，以复杂的方式组合在一起，而构建完成这个 model 则可能需要数天的时间。你不能保证自己能精确地记得每个函数是干什么的吧？所以，写上 DocString 会节省你自己和(阅读你代码的)其他人的时间。

在模型构建完成之后，和数据一样，同样应该对模型进行检视。将模型打印出来，或者使用其他的可视化工具(tensorboard可以, wandb在 file 里面可以检视 onnx 形式保存的模型)，检查模型的结构，看看是不是符合预期。

还有, 使用 model profiler 看看模型的参数量是多少(这个 pl 好像会进行提示，包括每一个 layer 和整个 model 的参数量)？FLOPS(如果需要的话，我还不太了解)？

## 训练过程中

定义好模型和数据之后，想要开始训练，依然有很长的一段路要走。

### ckpt

训练过程中保存 ckpt 是必须的。问题在于，如何优雅的保存 ckpt, 并且在程序崩溃的时候从最优的 ckpt 恢复. 已知 pl 有 ModelCheckpoint 的 callback, 能够指定很多参数：依据什么 metric 保存， 保存 top 几的 ckpt, ckpt 的保存路径, ckpt 的命名方式，等等。常规来说，功能已经够用。如果中间机器不宕机，只需要让训练一直持续下去，然后在一段时间之后(比如 24 小时)停止训练，并且取出 ckpt, 当做最终的结果。

当然，还有额外的情况：机器宕机了，怎么让程序自己重新启动并从最优的 ckpt 恢复？(比如说我正在吃饭……) 这个要求似乎有点吹毛求疵，而且做起来难度应该挺大，所以可以留待以后思考。目前的方案仅仅是，因为可以使用第三方工具(for me, wandb)检视训练过程, 所以如果我把程序丢进去训练，我依然能够在远端看到训练的情况(同步是实时的)。如果我经常检查训练状况的话，程序宕机了我会知道。

### log

这里主要指的是对训练过程中 loss, acc, 等各种 metric 的 log. 在 pl 里面，使用方法也非常的简单，这里就不在赘述了，直接去官网上看例子就行。

使用第三方的检视工具，需要在 Trainer 中指定 logger. wandb watch model 可以记录模型的 weight 和 grads, 这非常有用。wandb 还支持一些特比的 log 方式(比如图片组)。

### hyper param

超参数的选择一直是一个难题。but, 可以使用 wandb sweep 寻找超参数。(流行的三种方法就是 grid, random, bayes. grid 就是爆搜，random 其实也是爆搜. 貌似还有其他 auto ML 的方法。)

快速上手的使用示例：

[Pytorch-Lightning with Weights & Biases.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW?usp=sharing#scrollTo=27DZPzx-zK8k)

使用文档：

[wandb.sweep - Documentation](https://docs.wandb.ai/ref/python/sweep)

[Sweep Configuration - Documentation (wandb.ai)](https://docs.wandb.ai/guides/sweeps/configuration#structure-of-the-sweep-configuration)

### callback

callback 可以说包含了 log, ckpt 等等功能，因为他们都是在模型训练主体之外完成的， 属于工程代码, callback 就是用来安装工程代码的地方。

目前我对自定义 callback 的使用，和一般应用还不是很熟悉，可以去 pl 的文档里面看。

### 其他的训练 trick

可以使用的 trick 很多。

[Training Tricks — PyTorch Lightning 1.4.0rc1 documentation (pytorch-lightning.readthedocs.io)](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#advanced-gpu-optimizations)

[Performance and Bottleneck Profiler — PyTorch Lightning 1.4.0rc1 documentation (pytorch-lightning.readthedocs.io)](https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html)

[Learning Rate Finder — PyTorch Lightning 1.4.0rc1 documentation (pytorch-lightning.readthedocs.io)](https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html)

[Early stopping — PyTorch Lightning 1.4.0rc1 documentation (pytorch-lightning.readthedocs.io)](https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html)



## config

对于配置项的管理。确定好上面的东西，就可以写 main, 把所有的东西整合到一起，开始训练了。但是要考虑到使用什么方式指定参数。哪些参数从命令行输入？哪些参数应该保存在特定的文件中？

对于 model 和 data 的参数，他们应该保存在文件中。yaml(或者 yml) 文件就是非常好的格式(JSON也可以)。使用简单的缩进来表示从属结构，表达能力和可读性都很强。

在命令行中指定一些其他的参数。比如，本次实验的名字，ckpt的路径，训练使用多少 gpu, 训练多少轮，配置文件的路径，等等。

对于这两种参数，又有不同的处理方式。

1.   文件中的参数直接使用 omegaconf 加载，非常方便。

2.   命令行参数使用 argparse 处理。在 main 函数前面调用 parse() 函数，而使用 parse() 函数定义接受的命令行参数，最终 parse() 函数返回用户输入的命令行参数。之后，可以整合进 omegaconf 中统一进行结构化管理。(argparse 解析得到的是一个 namespace, 可以用 `vars(args)` 转成 dict, 然后传到 OmegaConf 的 create 或者 merge method 里面)

## 工具？

有没有这样一种工具(或者说，有机会自己开发这样一种工具): 自己定义好模型， data, 训练的 pipeline 之后，留出一些接口，比如模型的一些超参数，可以选择的 callbacks, 保存 ckpt 的路径和命名方式，等等。 这个工具就是简单地创建一个 web 页面，不过可以在其他设备上访问。 为了简洁，界面可以就是一些简单的 title 和下拉框，选择好之后，点击 start, 发信息给远端的服务器， 开始训练，然后页面将你引导到第三方 log 工具的页面，来检视模型的训练过程。

不过这个应用好像有些冗余，而且应该是给外行使用的。而且如果要真正给外行使用，还需要更多的可选项(比如可以选择的模型，使用什么数据，等等)。上面的要求使用笔记本电脑 + 远程服务器就可以做到，而且能做的更加灵活。