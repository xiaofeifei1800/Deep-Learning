#coding=utf-8
import numpy
import theano
import theano.tensor as T


rng = numpy.random

# get 10 random sample with 3 features
N = 10
feats = 3
D = (rng.randn(N, feats).astype(numpy.float32), rng.randint(size=N, low=0, high=2).astype(numpy.float32))


# set explotory variable and respones variable
x = T.matrix("x")
y = T.vector("y")

# set parameters
w = theano.shared(rng.randn(feats), name="w")

b = theano.shared(0., name="b")

# #构造损失函数
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # s激活函数
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # 交叉商损失函数
cost = xent.mean() + 0.01 * (w ** 2).sum()# 损失函数的平均值+L2正则项，其中权重衰减系数为0.01
gw, gb = T.grad(cost, [w, b])             #对总损失函数求参数的偏导数

prediction = p_1 > 0.5                    # 预测

train = theano.function(inputs=[x,y],outputs=[prediction, xent],updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))#训练所需函数
predict = theano.function(inputs=[x], outputs=prediction)#测试阶段函数

#训练
training_steps = 1000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
