#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azure.storage.file import FileService
from azure.storage.file import ContentSettings

import cv2
import numpy as np


# In[2]:


file_service = FileService(account_name='covidmodels', account_key='')


# In[ ]:





# In[3]:


data = []
labels = []


# In[4]:


import os
kaggle_data = os.listdir('./kaggle_data/')
azure_data = file_service.list_directories_and_files('covid-share/data')

for i, file_or_dir in enumerate(azure_data):
    print(f'COIVD - {file_or_dir.name} | NORMAL - {kaggle_data[i]}')
    
    # Getting file from storage
    byte_data = file_service.get_file_to_bytes('covid-share', 
                                               'data', 
                                               file_or_dir.name).content
    np_bytes = np.fromstring(byte_data, np.uint8)
    
    # Reshape
    az_img = cv2.imdecode(np_bytes, cv2.COLOR_BGR2RGB)
    az_img = cv2.resize(az_img, (224, 224))
    
    data.append(az_img)
    labels.append(1)
    
    kag_img = cv2.imread('./kaggle_data/' + kaggle_data[i])
    kag_img = cv2.cvtColor(kag_img, cv2.COLOR_BGR2RGB)
    kag_img = cv2.resize(kag_img, (224, 224))

    data.append(kag_img)
    labels.append(0)
    
#     break


# In[5]:


for i, d in enumerate(data):
    if len(d.shape) == 2:
        x = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        data[i] = x

data = np.array(data).astype('float')
labels = np.array(labels).astype('float')

print('Data:', data.shape)
print('Labels', labels.shape)


# In[6]:


data = data / 255


# In[7]:


# import pickle

# pickle.dump(data, open('data.pkl', 'wb'))
# pickle.dump(labels, open('labels.pkl', 'wb'))


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size=0.1, 
                                                    random_state=42)


# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[11]:


INIT_LR = 1e-3
EPOCHS = 4
BS = 10


# In[12]:


baseModel = VGG16(weights="imagenet", include_top=False,
                    input_tensor=Input(shape=(224, 224, 3)))


# In[13]:


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)


# In[14]:


model = Model(inputs=baseModel.input, outputs=headModel)


# In[15]:


for layer in baseModel.layers:
    layer.trainable = False


# In[16]:


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,
                metrics=["accuracy"])


# In[19]:


print(X_train.shape)
print(y_train.shape)


# In[21]:


H = model.fit(X_train, y_train, batch_size=BS,
        validation_split=0.1,
        epochs=EPOCHS)


# In[23]:


predIdxs = model.predict(X_test, batch_size=BS)
predIdxs = np.where(predIdxs < 0.5, 0, 1)
predIdxs


# In[25]:


cm = confusion_matrix(y_test, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])


# In[26]:


print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))


# In[29]:


from tensorflow.keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)


# In[30]:


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


# In[55]:


frozen_graph = freeze_session(tf.keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])


# In[67]:


from datetime import datetime
timestamp = str(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
model_name = 'tf_model_' + timestamp + '.pb'
model_name
path = tf.train.write_graph(frozen_graph, ".", model_name, as_text=False)


# In[68]:


print(f'Model saved to {path}')


# In[69]:


azure_models = file_service.list_directories_and_files('covid-share/model')

for file in azure_models:
    print(f'Deleting {file.name}')
    file_service.delete_file('covid-share', 'model', file.name)


# In[70]:


def upload_to_azure(filename):
	print(f'Uploading {filename}')

	file_service.create_file_from_path(
	    'covid-share',
	    'model', 
	    filename,
	    filename)


# In[71]:


upload_to_azure(path[2:])


# In[ ]:

print('Restarting API container')
import requests

resp = requests.post('https://management.azure.com/subscriptions/09df3e2b-c25c-4749-a42d-b5e91c715fa7/resourceGroups/cont/providers/Microsoft.ContainerInstance/containerGroups/covid/restart?api-version=2018-10-01')
