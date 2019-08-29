import pickle

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files

#尺寸
p_size = 100
# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    image = np.array(img)
    return image


# 测试图片
def evaluate_one_image(image_array, N_CLASSES):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, p_size, p_size, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[p_size, p_size, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'D:/myself/recognitionData/savedata'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            # 打印输出所有标签的得分
            print(prediction)
            # 得到所对应的标签
            max_index = np.argmax(prediction)
            print(prediction[0][max_index])
            # 从label_to_name_dic文件中提取出信息并根据对应下标找到人名
            with open(logs_train_dir + "/label_to_name_dic", 'rb') as o:
                data = pickle.load(o)
            print(data)
            return data[max_index]


# ------------------------------------------------------------------------

if __name__ == '__main__':
    img = Image.open('D:/myself/recognitionData/test/wwa.jpg')
    plt.imshow(img)
    plt.show()
    imag = img.resize([p_size, p_size])
    image = np.array(imag)
    # 输入图片和已经训练好的分类有几类
    print(evaluate_one_image(image, N_CLASSES=4))
