import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import numpy as np
from tensorflow.keras import regularizers

class animal10_classifier_ori(object):
    def __init__(self,img_size,batch_size):
        self.train_data_generator  =ImageDataGenerator(rescale=1.0/255.0,
                                                       rotation_range=5,
                                                       width_shift_range=0.02,
                                                       height_shift_range=0.02,
                                                       shear_range=0.02,
                                                       horizontal_flip=True,
                                                       vertical_flip=True)
        self.test_data_generator  =ImageDataGenerator(rescale=1.0/255.0,
                                                      rotation_range=5,
                                                      width_shift_range=0.02,
                                                      height_shift_range=0.02,
                                                      shear_range=0.02,
                                                      horizontal_flip=True,
                                                      vertical_flip=True)

        self.train_data_dir = './data/animal10/raw-img/train'
        self.test_data_dir = './data/animal10/raw-img/test'

        self.img_size = img_size
        self.batch_size =batch_size

        self.label_category_dic = {0: "dog",
                           1: "horse",
                           2: "elephant",
                           3: "butterfly",
                           4: "chicken",
                           5: "cat",
                           6: "cow",
                           7: "sheep",
                           8:'spider',
                           9: "squirrel",
                           }

    def get_local_data(self):
        train_data_iteratable =self.train_data_generator.flow_from_directory(self.train_data_dir,
                                                                      target_size=(self.img_size,self.img_size),
                                                                      batch_size=self.batch_size,
                                                                      class_mode='binary',
                                                                      shuffle=True,
                                                                      save_to_dir ='./data/changed_imgs')
        test_data_iteratable =self.test_data_generator.flow_from_directory(self.test_data_dir,
                                                                      target_size=(self.img_size,self.img_size),
                                                                      batch_size=self.batch_size,
                                                                      class_mode='binary',
                                                                      shuffle=True)

        return train_data_iteratable,test_data_iteratable

    def modle_gen(self):

        model = keras.Sequential([
            keras.layers.Conv2D(32,kernel_size=5,strides=1,padding='same',data_format='channels_last',activation=tf.nn.relu),
            keras.layers.MaxPool2D(pool_size=2,strides=2),
            keras.layers.Conv2D(32,kernel_size=6,strides=1,padding='same',data_format='channels_last',activation=tf.nn.relu),
            keras.layers.MaxPool2D(pool_size=2,strides=2),
            # keras.layers.Conv2D(128,kernel_size=7,strides=1,padding='same',data_format='channels_last',activation=tf.nn.relu),
            # keras.layers.MaxPool2D(pool_size=2,strides=2),
            # keras.layers.Conv2D(128,kernel_size=8,strides=1,padding='same',data_format='channels_last',activation=tf.nn.tanh ),
            # keras.layers.MaxPool2D(pool_size=2,strides=2),
            # keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(1024,kernel_regularizer =regularizers.l2(0.001),activation=tf.nn.relu),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(1024,kernel_regularizer =regularizers.l2(0.001),activation=tf.nn.relu),
            # keras.layers.Dropout(0.2),
            # keras.layers.Dense(256,kernel_regularizer =regularizers.l2(0.001),activation=tf.nn.relu),
            keras.layers.Dense(10,activation=tf.nn.softmax),
        ])

        return model

    def compile(self,model):

        model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.0001),
                      loss = keras.losses.sparse_categorical_crossentropy,
                      metrics =['accuracy'])

    def fit_model(self,model,train_data_iteratable,test_data_iteratable):

        modelckpt = keras.callbacks.ModelCheckpoint('./data/ckpt/original_model/weights_{epoch:02d}-{val_accuracy:.2f}.h5',
                                                    monitor='val_accuracy',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    mode='auto',
                                                    period=1
                                                    )

        model.fit_generator(train_data_iteratable,epochs=10,validation_data=test_data_iteratable,callbacks=[modelckpt])

    def predict(self,img_path,model):
        raw_img = load_img(img_path,target_size=(self.img_size*self.img_size))
        array_img = img_to_array().reshape(1,self.img_size,self.img_size,3)
        prediction = model.predict(array_img)
        index = np.argmax(prediction,axis=1)
        print(self.label_category_dic[index[0]])




if __name__ == '__main__':
    classifier  = animal10_classifier_ori(180,180)
    train_data_iteratable,test_data_iteratable =classifier.get_local_data()
    model =classifier.modle_gen()

    classifier.compile(model)
    classifier.fit_model(model,train_data_iteratable,test_data_iteratable)
