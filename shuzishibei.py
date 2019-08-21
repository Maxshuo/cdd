import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 50
n_batch = mnist.train.num_examples//batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,200]))
b = tf.Variable(tf.zeros([200]))

W1 = tf.Variable(tf.zeros([200,10]))
b1 = tf.Variable(tf.zeros([10]))

L1 = tf.nn.tanh(tf.matmul(x,W)+b)
prediction = tf.nn.softmax(tf.matmul(L1,W1)+b1)

tf.global_variables_initializer().run()

for i in range(2000):
    batch = mnist.train.next_batch(50) // 获取50条数据作为一个batch，这里可以叫做mini - batch

    train_step.run({x: batch[0], y_: batch[1], keep_prob: 0.5}) // 训练

    // 每100个batch的训练做一个预测
    if i % 100 == 0
        print("step：", str(i), "测试准确率：",
              accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) // eval()
        是tf.Tensor的Session.run()
        的另一种写法，但是eval()
        只能用于tf.Tensor类对象，即必须是有输出的operation，对于没有输出的operation，可以用run()
        或Session.run()
        即可。

