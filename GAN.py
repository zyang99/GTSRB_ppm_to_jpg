import tensorflow as tf
from PIL import Image
import numpy as np


#导入所需要的数据集
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.
x_train = tf.reshape(x_train,[-1,28,28,1])
x_test = tf.reshape(x_test,[-1,28,28,1])
print(x_train.shape)
print(x_test.shape)

batch_size = 64
#构建输入到模型中的数据集
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size*5).batch(batch_size=batch_size)

#构建生成器模型
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(512)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(1024)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.fc4 = tf.keras.layers.Dense(784)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.bn1(x))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.bn2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.bn3(x))
        x = tf.nn.relu(self.fc4(x))
        return x

#构建判别器模型
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(512)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(1024)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.fc4 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.bn1(x))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.bn2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.bn3(x))
        x = tf.nn.relu(self.fc4(x))
        return x
#生成器的损失函数
def g_loss(generator,discriminator,noise):
    fake_image = generator(noise)
    d_fake_logits = discriminator(fake_image)
    y = tf.ones_like(d_fake_logits)
    loss = tf.keras.losses.binary_crossentropy(d_fake_logits,y)
    return tf.reduce_mean(loss)

#判别器的损失函数
def d_loss(generator,discriminator,noise,train_data):
    fake_image = generator(noise)
    d_g_logit = discriminator(fake_image)
    g_y = tf.zeros_like(d_g_logit)
    loss_fake = tf.keras.losses.binary_crossentropy(d_g_logit, g_y)
    g_r_logit = discriminator(train_data)
    r_y = tf.ones_like(g_r_logit)
    loss_real = tf.keras.losses.binary_crossentropy(g_r_logit, r_y)
    return tf.reduce_mean(loss_fake + loss_real)

#开始构建训练部分
g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

x = next(iter(train_db))
x = tf.reshape(x,[-1,28,28,1])
z = tf.random.normal([64,100])
g = Generator()
d = Discriminator()



#保存图片部分
def save_images(imgs,name):
    new_im = Image.new('L',(64,64))
    index = 0
    for i in range(0,64,8):
        for j in range(0,64,8):
            im = imgs[index]
            im = Image.fromarray(im,mode='L')
            new_im.paste(im,(i,j))
            index += 1
    new_im.save(name)


#开始进行网络训练
for epoch in range (1000):
    with tf.GradientTape() as tape:
        loss_f = g_loss(g,d,z)
    grads = tape.gradient(loss_f,g.trainable_variables)
    g_optimizer.apply_gradients(zip(grads,g.trainable_variables))
    with tf.GradientTape() as tape:
        loss_r = d_loss(g,d,z,x)
    grads = tape.gradient(loss_r, d.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, d.trainable_variables))
    print(epoch, 'loss_f:', float(loss_f), 'loss_r:', float(loss_r))
    z = tf.random.normal((64, 100))
    logits = g(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'E:\\GTSRB_ppm_to_jpg\\epoch_%d_sampled.png' % epoch)