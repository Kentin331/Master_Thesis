import pandas
import numpy as np
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Model
from statistics import mean
from scipy import stats
from numpy import std
from keras.utils import to_categorical
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#***************************** NEW CODE *********************************

def arrange_images(data_dir,dest_dir):
    label = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
    label_images = []
    # Copy the images into new folder structure:
    for i in label:
        os.mkdir(dest_dir + str(i) + "\\")
        sample = metadata[metadata['dx'] == i]['image_id']
        label_images.extend(sample)
        for id in label_images:
            shutil.copyfile((data_dir + "\\"+ id +".jpg"), (dest_dir + i + "\\"+id+".jpg"))
        label_images=[] 


#**************** CLASS WEIGHTS to Overcome Class Imbalance ********************
def estimate_weights_mfb(label):
    class_weights = np.zeros_like(label, dtype=np.float)
    counts = np.zeros_like(label)
    for i,l in enumerate(label):
        counts[i] = metadata[metadata['dx']==str(l)]['dx'].value_counts()[0]
    counts = counts.astype(np.float)
    median_freq = np.median(counts)
    for i, label in enumerate(label):
        class_weights[i] = median_freq / counts[i]
    return class_weights



# importing metadata

# Location of this program 
#os.chdir("E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code")
os.chdir("D:\\Ecole\\NTNU\\S2\\Implementation_Quentin")
data_dir = os.getcwd() + "\\Dataset\\HAM_Dataset"
metadata = pandas.read_csv(data_dir + '\\HAM10000_metadata',sep='\t', lineterminator='\r', header=0)

# label encoding the seven classes for skin cancers
le = LabelEncoder()
le.fit(metadata['dx'])
LabelEncoder()
print("Classes:", list(le.classes_))
# Adding the different classes to the dataset
metadata['label'] = le.transform(metadata["dx"]) 

# A path to the folder where the rearranged images are stored:
# as rearranged, it means : 
# - HAM10K
#   -ClassA
#       -images
#   -ClassB
#       -images
#   .
#   .
#   .
dest_dir = os.getcwd() + "\\Dataset\\HAM10K\\"

#arrange_images(data_dir,dest_dir)


label = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
classes = [ 'actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions', 
           'dermatofibroma','melanoma', 'melanocytic nevi', 'vascular lesions']



classweight= estimate_weights_mfb(label)

#different parameters for the model
batch_size = 3
nb_epochs = 12


#**************** Dataset Creation ********************

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

val_datagen = ImageDataGenerator(
    rescale=1./255.,
    validation_split=0.2
)

train_ds = train_datagen.flow_from_directory(
    dest_dir,
    target_size=(224,224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=None,
    subset="training",
    interpolation="bilinear",
    follow_links=False)

val_ds = val_datagen.flow_from_directory(
    dest_dir,
    target_size=(224,224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=None,
    subset="validation",
    interpolation="bilinear",
    follow_links=False)

class_names = train_ds.class_indices
print(class_names)

#Little plot to show some images and the corresponding class
plt.close()
im,lab = train_ds.next()
plt.imshow(im[0])
plt.title(lab)
#plt.show()
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     sample = train_ds.next()
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(sample.numpy().astype(np.uint8))
#     #plt.title(sample)
#     plt.axis("off")
# plt.show()

#**************** Model Creation (import the Inception V3 and perform transfer learning) ********************



pre_trained_model = ResNet101(input_shape = (224,224,3), include_top= False, weights = "imagenet")

for layer in pre_trained_model.layers:
    layer.trainable = False

x = layers.Flatten()(pre_trained_model.output)

x = layers.Dense(1024, activation = 'relu')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(7,activation='sigmoid')(x)

model = Model(pre_trained_model.input,x)

model.compile(optimizer = Adam(lr = 0.001),loss = "categorical_crossentropy", metrics = ['acc']
)


hitsory = model.fit_generator(
    train_ds,
    steps_per_epoch = train_ds.samples // batch_size,
    validation_data = val_ds,
    validation_steps = val_ds.samples // batch_size,
    epochs = nb_epochs,
    verbose =2,

)