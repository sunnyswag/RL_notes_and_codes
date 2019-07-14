# RL_notes_and_codes

* 哎，readme.md里面的latex公式无法在线渲染，本地查看是没毛病的

## 1. 学习笔记包括

**RL**：

* [berkeley_CS294-112(2018)](./berkeley_CS294-112)

  [课程主页](http://rail.eecs.berkeley.edu/deeprlcourse/)

  [课程视频](https://www.youtube.com/playlist?list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37)

* [stanford_cs234_winter(2019)](./stanford_cs234_winter)

  [课程主页](https://web.stanford.edu/class/cs234/schedule.html)

  [课程视频](https://www.youtube.com/watch?v=FgzM3zpZ55o&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

* [RL Course by David Silver](./RL_by_David_Silver) ([slides](./RL_by_David_Silver/slides))

  [课程主页](<http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html>)

  [课程视频](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)

* [经典算法阅读](./algorithm_reading)

* [日常的困惑记录](./other_notes.md)

**DL**：

* [Spinning Up as a Deep RL Researcher](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#the-right-background)

  之前看的《[动手学深度学习](<https://zh.gluon.ai/>)》(没看完，边敲边看看到了优化算法)，也有些遗忘，准备通过这个复习一遍，感觉系统的学还要等到系统学完机器学习吧。笔记[在此](./deep_learning/Spinning_Up_as_a_Deep_RL_Researcher.md)

**ML**：

* [Machine_Learning_Foundations](./Machine_Learning_Foundations)

## 2.强化学习资料

* ##### 教程：

  [RL Course by David Silver](<http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html>)无痛入门，建议看youtube视频(B站视频画质感人)。虽然没字幕，但都是日常用语，查了几个关键字之后就能听懂了

  [李宏毅的强化学习教程](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)，讲的有点宽泛，有些公式有点难理解，可以当作科普

  [一位法国小哥的强化学习教程](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)，讲的由浅入深，很适合入门。但是用tensorflow实现的，当时虽然对着他的代码敲了一遍。没系统学tensorflow，对训练时的session没有仔细理解。算是通过这个教程把基本的原理理解了吧。期间提到了一个policy gradient的[教程](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)，但实在公式太多了，而且很长，一直没去看。所以现在(2019/6/15)对policy gradient不是特别理解。大致理解为: 输入state，直接输出action，而不是输出值，选择Q值最大action。

  [morvanzhou的强化学习教程](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)，个人来说不是很推荐，因为他把Q-learning讲的不是很清楚（似乎是讲错了？），所以就没往下看了，不过还是很感谢他！

  [MIT 6.S191: Introduction to Deep Learning](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)，不知道怎么样，听说评价还挺好的，有时间去看看

* ##### 书籍

  [Reinforcement Learning: An Introduciton](http://incompleteideas.net/book/bookdraft2018jan1.pdf)，我买的是某宝上的打印版，英文原版也太贵了吧(好像已经下架了)。真心希望他不要吃灰把，不过英文版确实有点难啃。

* ##### Github_repo

  收集的一些有意思和star数很高的github仓库

  [openai/gym](https://github.com/openai/gym)，可以试着玩一下，star数很高，支持一些游戏和经典算法

  [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)，unity提供的环境，感觉会很有意思

  [deepmind/lab](https://github.com/deepmind/lab)，鼎鼎有名的deepmind提供的3D强化学习环境

  [openai/retro](https://github.com/openai/retro/tree/develop)，openai提供的retro游戏，似乎多到你无法想象

  [google/dopamine](https://github.com/google/dopamine)，谷歌的多巴胺强化学习框架，个人对框架不是很感冒

* ##### 论文

  知乎有大佬整理的[论文合集](https://zhuanlan.zhihu.com/p/23600620)，但我觉得我一辈子也看不完这么多论文的

* ##### 国内研究强化学习的公司

  或许学了不想学的时候就点开这些网页看看，给自己打打气吧

  [启元世界](http://www.inspirai.com/index.html)

  [深极智能 – 用人工智能改变游戏业](http://www.levelup-ai.cn/)

  [网易游戏伏羲人工智能实验室招聘--期待与你一起点亮游戏未来！](https://fuxi.163.com/index.html)

* ##### 名校查询

  在[这里](http://csrankings.org/#/fromyear/2017/toyear/2019/index?ai&canada)你可以看到全世界研究强化学习的一些高校，当然不止强化学习的查询

  想想自己以后会去哪里呢，感觉加拿大是个不错的选择吧



## 3.Logs

​	Today(2019/6/12)，被CS234 Lecture2的notes和Assignments所吓到，准备老老实实看[RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)了，江湖有缘再见吧

​	Today(2019/7/6)，到现在还有第九章的后一段和第十章没看，感觉已经差不多了，后边都是应用的，而且第十章的PPT也非常不清晰。由于我是直接在youtube上看的，没有字幕，确实有一点点难受。感觉讲的都比较基础把，所以用到了很多公式，在letax公式方面，自己也学到了一些。这门课相当于是鼻祖，所以会比其他课讲的都要好(个人感觉,因为其他课程也是在这门课上有所改进和删减)。算是把基础给通关了，之后的话自己可以根据别人的思路来写一些代码，可以从最简单最基础的写起。如果突然又想补补基础了，就把书或者berkeley的教程看一遍，加油鸭

​	Today(2019/7/13)，实现了Q-learning,policy gradient, DQN,准备开始学习ML。完成其中的作业并做一些kaggle的实战。