import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import itertools

# all images will be converted to this size
ROWS = 256
COLS = 256
CHANNELS = 3

train_image_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, rotation_range=45)
test_image_generator = ImageDataGenerator(horizontal_flip=False, rescale=1./255, rotation_range=0)

train_generator = train_image_generator.flow_from_directory('train', target_size=(ROWS, COLS), class_mode='categorical')
test_generator = test_image_generator.flow_from_directory('test', target_size=(ROWS, COLS), class_mode='categorical')

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(1024, activation='relu')(x)
out_layer = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=out_layer)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

tensorboard = TensorBoard(log_dir='./logs/inceptionv3')

model.fit_generator(train_generator, steps_per_epoch=32, epochs=100, callbacks=[tensorboard], verbose=2)

print(model.evaluate_generator(test_generator, steps=5000))

# unfreeze all layers for more training
for layer in model.layers:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=32, epochs=100)

test_generator.reset()
print(model.evaluate_generator(test_generator, steps=5000))

model.save("birds-inceptionv3.model")



from keras.models import load_model
from keras.preprocessing import image
from os import listdir
import numpy as np

ROWS = 256
COLS = 256

CLASS_NAMES = sorted(listdir('images'))

model = load_model('birds-inceptionv3.model')

def predict(fname):
    img = image.load_img(fname, target_size=(ROWS, COLS))
    img_tensor = image.img_to_array(img) # (height, width, channels)
    # (1, height, width, channels), add a dimension because the model expects this shape:
    # (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0) 
    img_tensor /= 255. # model expects values in the range [0, 1]
    prediction = model.predict(img_tensor)[0]
    best_score_index = np.argmax(prediction)
    bird = CLASS_NAMES[best_score_index] # retrieve original class name
    print("Prediction: %s (%.2f%%)" % (bird, 100*prediction[best_score_index]))

predict('test-birds/annas_hummingbird_sim_1.jpg')
predict('test-birds/house_wren.jpg')
predict('test-birds/canada_goose_1.jpg')

# interactive user input
while True:
    fname = input("Enter filename: ")
    if(len(fname) > 0):
        try:
            predict(fname)
        except Exception as e:
            print("Error loading image: %s" % e)
    else:
        break