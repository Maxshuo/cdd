import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples//batch_size

#// weight_variable()方法用于初始化权重，以便之后调用，这里用截断正太分布进行初始化，随机加入噪声。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#// bias_variable()用于初始化偏置，初始化为常量0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#// 定义卷积操作，步长为1×1，padding='SAME'会在输入矩阵上下各加几个值为0的行，在左右各加个几个值为0的列，以提取边缘特征，意味着卷积后图像的大小为原尺寸/步长，显然横向除以横向，纵向除以纵向。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#// 定义池化操作，这里定义最大池化，即在区域里提取最显著的特征。池化核的大小为2×2。池化后图像的大小为原尺寸/步长。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])#// x是特征，即输入的数据，是一个长度为784的数组，None代表行数不确定
x_image = tf.reshape(x, [-1, 28, 28, 1])#// reshape()将读入的x输入变成图像的格式，即28*28，以便之后处理,-1和None相对应，即图片的数量不确定，1代表灰度图像。即输入为28*28*1的x_image。
y = tf.placeholder(tf.float32, [None, 10])#// y_是真实的输出label，是一个长度为10的数组

W_conv1 = weight_variable([5, 5, 1, 32])#// 卷积核的为5*5，单通道的灰度图像，一共32个不同的卷积核，输出32个特征图
b_conv1 = bias_variable([32])#// 每个卷积对应一个偏置，共32个偏置
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)#// 先进行卷积操作，后用relu激活函数处理，得到处理后的输出28*28*32。
h_pool1 = max_pool_2x2(h_conv1)#// 进行最大池化，28*28*32变成了14*14*32

W_conv2 = weight_variable([5, 5, 32, 64])#// 卷积核尺寸为5*5,32个通道，因为上面得到的是32层，第二个卷积层有64个不同的卷积核
b_conv2 = bias_variable([64])#//64个偏置
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)#// 第一层卷积和池化后的输出h_pool1作为第二层卷积层的输入，由于步长strides为1，填充边界padding为1，故结果为14*14*64
h_pool2 = max_pool_2x2(h_conv2)#// 由于步长为2，padding='SAME'，故最大池化后为7*7*64

W_fc1 = weight_variable([7*7*64, 100])#// 第一个全连接层的权重矩阵，输入是上层的节点数，上层是7*7*64个结点，输出是1024个结点，1024是自己定义的。
b_fc1 = bias_variable([100])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#// 为方便全连接层计算，把上一层传过来的结果reshape，从三维的变成一维的。
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)#// 得到一个长度为1024的输出

W_fc2 = weight_variable([100, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2)+b_fc2)#// 使用softmax作为激活函数,softmax是一个多分类器。一般配合交叉熵损失函数使用

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss) #// 优化器使用Adam，学习率用一个较小的1e-4

init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y,axis=1),tf.argmax(prediction,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print(str(epoch)+" : "+str(acc))