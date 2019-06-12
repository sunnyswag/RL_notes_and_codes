# RL_notes_and_codes
## 1. 学习笔记包括

* [berkeley_CS294-112(2018)](./berkeley_CS294-112)

  [课程主页](http://rail.eecs.berkeley.edu/deeprlcourse/)

  [课程视频](https://www.youtube.com/playlist?list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37)

* [stanford_cs234_winter(2019)](./stanford_cs234_winter)

  [课程主页](https://web.stanford.edu/class/cs234/schedule.html)

  [课程视频](https://www.youtube.com/watch?v=FgzM3zpZ55o&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

## 2.强化学习资料

* ##### 教程：

  [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)，[slides](<http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html>)由于时间有点久远，至今没有看

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

​	今天(2019/6/12)，被CS234 Lecture2的notes和Assignments所吓到，准备老老实实看[RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)了，江湖有缘再见吧
