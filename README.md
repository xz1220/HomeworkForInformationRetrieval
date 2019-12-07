<h1 style="text-align: center"> 基于矩阵分解算法的评分预测实现 </h1>
<div style="text-align: center"><big>邢政     2017060801025</big></div>
<div style="text-align: center"><small>12 - 7 - 2019</small></div>


___

>协同过滤在推荐系统领域有着广泛的应用。矩阵分解算法是其中具有代表性的算法之一。在本次课程作业中，针对于ml-1m数据集，我们利用了pandas进行数据的读取和划分，8/10为训练集，1/10为验证集，1/10为测试集。并且，我们使用keras实现了一个简单的神经网络模型来拟合一个矩阵分解算法。由于数据集较小的缘故，我们使用了早停的方式来防止模型对数据集过拟合。在训练集上，模型的loss降低至**0.49**，在验证集和测试集上，模型的loss也只有**0.75**。


## Introduction

 在这个部分主要是介绍我们所使用的数据集以及我们的算法的原理。代码上传在了github上，请点击[传送门](https://github.com/xz1220/HomeworkForInformationRetrieval)。

#### Movielens Datastes

movielens 数据集收集了许多人关于不同电影的评价。这次我们所使用的是其中一个版本，有**1000209**条评价记录。
数据集总共有分为三个部分——users、movies和rating。

- users
  - Gender ：表示性别。M代表男性，F代表女性。
  - Age ：1 表示小于18岁；18 意味着年龄在 18~24 之间；25 意味着年龄在 25~34 之间；以此类推。
  - Occupation ：表示职业。

- movies
  - Title ：代表电影的名字，包含有电影发售的年份。
  - Genres ：代表电影的类别。

- rating
  - UserId ：代表用户的唯一标识号。
  - MoviesId :代表电影的唯一标识号。
  - Rating: 表达了一个用户对于一部电影的喜恶程度。

对于本次课程作业来说，有用的数据只有rating文件夹里面的UserId、MoviesId以及Rating。我们需要做的就是将这三列数据提取出来并且各自做处理。

#### Matrix Factorization
矩阵分解算法是协同过滤算法的一种。它在2006年获得了Netflix推荐大赛的奖项，在整个推荐系统发展史上具有举足轻重的地位，对促进推荐系统的大规模发展及工业应用功不可没。
##### 核心思想
一个用户的操作行为可以转化为行为矩阵$R \in R^{n \times n}$，其中$R_{i,j}$代表了用户i对物品j的评分。矩阵分解算法就是将用户评分矩阵R分解为两个矩阵$U_{n \times k}$和$V_{k \times m}$的乘积。
$$
R_{n \times n} = U_{n \times k} \times V_{k \times m}
$$
其中，$U_{n \times k}$代表用户特征矩阵，$V_{k \times m}$代表的是物体特征矩阵。而某个用户对于某个目标物体的评分可以通过提取用户特征矩阵$U_{n \times k}$对应的行和物品特征矩阵$V_{k \times m}$对应的列相乘来得到。

矩阵分解的目的是通过机器学习的手段将用户行为矩阵中缺失的数据(用户没有评分的元素)填补完整，最终达到可以为用户做推荐的目标。

#### Neural Network
大名鼎鼎的神经网络，深度学习的基石之一，它的原理我就不再赘述了。

我要提的一点是：
>**万能近似定理**:只要隐含层的神经元足够多，一个两层的神经网络就可以你和任意复杂度的函数。


## Method
本次实验所使用的方法来自于《Deep Matrix Factorization Models for Recommender Systems》这篇文章。是作者根据神经网络提出的一种新的矩阵分解模型，发表在2017年的IJCAI上。
<img height="300px" src="https://github.com/xz1220/HomeworkForInformationRetrieval/blob/master/论文模型结构图.jpg"/>

#### Data Processing

#### Model








## Text Formating

Regular, **bold**, *italic*, ~~strike~~, ==hightlight==, `inline-code`,*emphasis^,<!--comment-->,

## Cites

> This is a cite

## Inline math:

Inline math $ X^2 + 1 = 1 $ works fine.

## Math Block:

$$
X^2 + 1 = 1
$$

## Tables:

| First Column | Second Column | Third Column |
| ------------ | ------------- | ------------ |
| One          | Two           | Three        |
| Four         | Five          | Six          |

## Code:

```js
import someCode from 'someLibrary';
```



## Lists

- First item
- Second item
  - Third item
    - Another level

## Links

[This is a link](www.google.com)

## Footnote

Some thing 

## Superscripts

Example^1^

Example~2~

## Images

<img height="300px" src="https://image.freepik.com/vector-gratis/garabatos-ciencia_23-2147501583.jpg"/>

## 






