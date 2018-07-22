# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing

df = pd.read_csv("wine.csv", header=0)
print (df.describe())
#读入数据集
for i in range (1,8):
    number = 420 + i
    ax1 = plt.subplot(number)
    ax1.locator_params(nbins=3)
    plt.title
    ax1.scatter(df[df.columns[i]],df['Wine']) #Plot a scatter draw of the  datapoints
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
sess = tf.InteractiveSession()

X = df[df.columns[1:13]].values
#读入输入项
y = df['Wine'].values-1
#读入输出项
Y = tf.one_hot(indices = y, depth=3, on_value = 1., off_value = 0., axis = 1 , name = "a").eval()
#将标签转化为一位热码
X, Y = shuffle (X, Y)
#打乱数据集

scaler = preprocessing.StandardScaler()
#计算训练集的平均值和标准差
X = scaler.fit_transform(X)
#对输入数据集采用相同的变换

# 创建模型
x = tf.placeholder(tf.float32, [None, 12])  #输入占位符
W = tf.Variable(tf.zeros([12, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 3])  #输入占位符
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))   #计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)     #用梯度下降法进行训练


# 训练神经网络
tf.initialize_all_variables().run()
for i in range(100):
  X,Y =shuffle (X, Y, random_state=1)

  Xtr=X[0:140,:]
  Ytr=Y[0:140,:]

  Xt=X[140:178,:]
  Yt=Y[140:178,:]
  Xtr, Ytr = shuffle (Xtr, Ytr, random_state=0)
  #batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs, batch_ys = Xtr , Ytr
  train_step.run({x: batch_xs, y_: batch_ys})
  cost = sess.run (cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
  # 测试训练好的模型
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(accuracy.eval({x: Xt, y_: Yt}))