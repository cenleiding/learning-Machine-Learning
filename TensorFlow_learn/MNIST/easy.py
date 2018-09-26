# coding:utf-8

"""
@author : CLD
@time:2018/9/2014:44
@description: 简单级
"""
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#模型构建
x=tf.placeholder('float',[None,784])
y_=tf.placeholder("float", [None,10])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
#cost :交叉熵
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#采用梯度下降
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化变量
init = tf.initialize_all_variables()
#创建session用于运行
sess=tf.Session()
sess.run(init)
#开始训练
for i in range(0,1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)                    #随机训练
    sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})       #将占位符替换掉
    #评估
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #预测是否和真实值相同
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  #转换 布尔值->数值
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()

