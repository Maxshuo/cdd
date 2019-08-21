import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise =np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

W1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1]))

WX1 = tf.matmul(x,W1)+b1
L1 = tf.nn.tanh(WX1)

WX2 = tf.matmul(L1,W2)+b2
prediction = tf.nn.tanh(WX2)

loss = tf.reduce_mean(tf.square(prediction-y_data))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
    prediction_vlaue = sess.run(prediction,feed_dict={x:x_data})
    print(sess.run(loss,feed_dict={x:x_data}))
plt.figure()
plt.scatter(x_data,y_data)
plt.plot(x_data,prediction_vlaue,'r-',lw=5)
plt.show()