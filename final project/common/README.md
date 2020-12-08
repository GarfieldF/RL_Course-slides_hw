# Common 类

`env`中包含一些常用的环境wrapper（参考了[openai-baseline](https://github.com/openai/baselines))

特别地：

* `env.FireResetEnv`: 有些游戏`只有按下`Fire`键后，游戏才会开始，因此需要在reset处加入Fire动作
* `env.ParallelEnv`: 为了避免样本之间的时间相关性，采用了多进程的实现方式
* `env.ClipRewardEnv`: 某些游戏的得分存在跳变：比如连续得分会加倍。这可能导致训练非常地不稳定，为了避免这样的情况，我们限制了reward的幅度
* `env.MaxAndSkipEnv`: 一个动作执行4帧
* `env.EpisodicLifeEnv`: 注意到有些游戏有生命值的概念，每次失败，只会损失生命值，而不会游戏结束。这里我们修改设定，使得损失生命值也会输出episode结束的信号
* `env.WrapFrame`: 借鉴DQN原文，使用了84x84的黑白图像作为输入
* `env.StackFrame`: 将4帧图像叠在一起，作为状态的输入

