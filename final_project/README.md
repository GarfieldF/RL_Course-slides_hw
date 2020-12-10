# Pytorch-RL

这里包含了一些使用pytorch对一些标准强化学习的实践。

和一些著名的实践不同（比如[openai-baseline](https://github.com/openai/baselines))，我们的实践每个算法相对比较独立，而且只关注了最核心的方向，适合理解。

## 运行环境

* [Gym[atari]](https://github.com/openai/gym)

* [Pytorch v0.4.1](https://pytorch.org/)
* Python 3.6

## 运行方式（以A2C为例）

运行需要进入到子目录中。

```shell
cd A2C
python run.py
```

### 测试

```shell
cd A2C
python test.py
```

> 注意到，在pretrained.pth存储了我所训练的模型（并未充分训练，但是已经能取得一些好的效果），可以用来测试效果。

使用pretrained的模型进行测试

```shell
python test.py --model_path ./pretrained.pth
```

这是A2C的结果

![A2C_Result](./A2C/Result.gif)

## 文件结构

`Common`中包含了常用的环境wrapper.

其他每个文件夹包含某一个特定的算法。其中可能会包含如下的代码：

* `model.py` 模型文件
* `run.py` 用于训练
* `test.py` 用于测试
* `a2c_agent.py` (以A2C为例)
    * `a2c_agent.ObsPreproc`预处理
    * `a2c_agent.A2CAgent`
    * `a2c_agent.TestAgent`测试agent
        * 要求环境单进程的（为了可视化）
        * 测试过程中会不断地更换随机种子
        * 一般，测试环境中不会对reward进行限制，可以统计真实得分
* …...

