from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from tensorflow.keras import regularizers


class animal10_classifier_base_VGG(object):

    def __init__(self, img_size, batch_size):
        self.train_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
        self.test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

        self.train_dir = "./data/animal10/raw-img/train"
        self.test_dir = "./data/animal10/raw-img/test"

        self.image_size = img_size
        self.batch_size = 32

        # 获取VGG的notop的模型，以进行模型扩展
        self.VGG_base_model = VGG16(weights='imagenet', include_top=False)

        self.label_dict = {'0': "dog",
                           "1": "horse",
                           "2": "elephant",
                           "3": "butterfly",
                           "4": "chicken",
                           "5": "cat",
                           "6": "cow",
                           "7": "sheep",
                           '8': 'spider',
                           "9": "squirrel",
                           }

    # 读取本地动物图片数据
    def get_local_data(self):
        train_data = self.train_data_gen.flow_from_directory(self.train_dir,
                                                             target_size=(self.image_size, self.image_size),
                                                             batch_size=self.batch_size,
                                                             class_mode='binary',
                                                             shuffle=True)
        test_data = self.test_data_gen.flow_from_directory(self.test_dir,
                                                           target_size=(self.image_size, self.image_size),
                                                           batch_size=self.batch_size,
                                                           class_mode='binary',
                                                           shuffle=True)
        return train_data, test_data

    def tune_base_model(self):
        x = self.VGG_base_model.outputs[0]

        # 全局池化以减少参数数量
        x = keras.layers.GlobalAveragePooling2D()(x)

        # 加入自己定义的神经网络
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        # x =keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_predict = keras.layers.Dense(10, activation=tf.nn.softmax)(x)

        # 得到新定义的模型
        tuned_model = keras.models.Model(inputs=self.VGG_base_model.inputs, outputs=y_predict)

        return tuned_model

    def freeze_model(self):
        """
        冻结VGG_notop模型里的参数
        """
        for layer in self.VGG_base_model.layers:
            layer.trainable = False

    def compile(self, model):
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return None

    def fit_generator(self, model, train_data, test_data):
        # 每迭代一次epoch记录validation_data测试中准确率最高的权重
        modelckpt = keras.callbacks.ModelCheckpoint(
            './data/ckpt/VGG_base_model/weights_{epoch:02d}-{val_accuracy:.2f}.h5',
            monitor='val_accuracy',
            save_weights_only=True,
            save_best_only=True,
            mode='auto',
            period=1)

        model.fit_generator(train_data, epochs=3, validation_data=test_data, callbacks=[modelckpt])

        return None

    # 用来识别具体的一张图片
    def predict(self, model, img_path):
        # 加载模型，transfer_model
        model.load_weights("./data/ckpt/VGG_base_model/weights_02-0.83.h5")

        # 读取图片，处理
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        img = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])

        # 得到预测结果的类别str
        predictions = model.predict(image)  # 得到概率序列
        indexes = np.argmax(predictions, axis=1)  # 得到概率最大的类别的编号
        print(self.label_dict[str(indexes[0])])

    # 用来计算模型分类准确率，循环100次，每次32个样本，一共3200个测试样本
    def accuracy_getter(self, model, test_data):
        model.load_weights("./data/ckpt/VGG_base_model/weights_02-0.83.h5")
        accuracy = np.sum((np.argmax(model.predict(test_data), axis=1) == test_data.labels) != 0) / \
                   test_data.labels.shape[0]
        print('根据{}个测试样本得出的测试准确率【测试正确样本数/样本总数】:{}'.format(test_data.labels.shape[0], accuracy))


if __name__ == '__main__':
    classifier = animal10_classifier_base_VGG(224, 32)

    # 获取数据并在VGG基础上扩展以适合本分类案例
    train_data, test_data = classifier.get_local_data()
    tuned_model = classifier.tune_base_model()

    # 用来训练模型
    # classifier.freeze_model()
    # classifier.compile(tuned_model)
    # classifier.fit_generator(tuned_model, train_data, test_data)

    # 用来识别一张图片，返回图片中文类型
    # img_path ='data/animal10/raw-img/test/pecora/e13cb60a2bf31c22d2524518b7444f92e37fe5d404b0144390f8c078a0eabd_640.jpg'
    # classifier.predict(tuned_model,img_path)
    # print(test_data[0])

    # 用于计算完成的模型的测试准确率
    # classifier.accuracy_getter(tuned_model,test_data)
