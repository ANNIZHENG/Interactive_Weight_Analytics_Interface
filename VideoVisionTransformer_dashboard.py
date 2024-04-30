#!/usr/bin/env python
# coding: utf-8

# ## Video Vision Transformer

# In[1]:


import cv2
import os
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import keras
from keras import layers, ops
import matplotlib.pyplot as plt
keras.utils.set_random_seed(42)
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True) 

from tensorflow.keras.models import Model
from sklearn.manifold import (TSNE)
from sklearn.metrics import roc_curve, auc
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import ipywidgets as widgets
import streamlit as st


# ## Parameters

# In[2]:


# `##` --> Adjustable

# DATA
IMG_SIZE = 128  ## Image size (128, 128) in this case
CHAN_SIZE = 1   # 1-GrayScale; 3-RGB
BATCH_SIZE = 8  ## 16, 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (50, IMG_SIZE, IMG_SIZE, CHAN_SIZE)
NUM_CLASSES = 2  # 1-Crash; 0-Normal

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
# EPOCHS = 10

# TUBELET EMBEDDING
PATCH_SIZE = (8,8,8) ##
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
EMBED_DIM = 64   ## Size of the feature vectors transformed from the input
NUM_HEADS =  6   ##
NUM_LAYERS = 6   ##


# ## Retrieve videos for training, validating, and testing

# In[3]:


# retrieve all video name

# frames_path = 'data/frames/'
# frames_path_normal = 'data/frames/Normal/'
# frames_path_crash = 'data/frames/Crash/'

frames_path = '../data/frames/'
frames_path_normal = '../data/frames/Normal/'
frames_path_crash = '../data/frames/Crash/'


frames_name_normal = sorted([f for f in os.listdir(frames_path_normal)])
frames_name_crash = sorted([f for f in os.listdir(frames_path_crash)])


# In[4]:


# how many data needed
num_normal = 50
num_crash = 50

frames_name_normal = random.sample(frames_name_normal, num_normal)
frames_name_crash = random.sample(frames_name_crash, num_crash)


# In[5]:


# 6:2:2.5 train test split

train_normal, test_normal = train_test_split(frames_name_normal,test_size=0.3, random_state=42)
train_crash, test_crash = train_test_split(frames_name_crash, test_size=0.3, random_state=42)

temporary_normal, test_normal = train_test_split(frames_name_normal, test_size=0.2, random_state=42)
temporary_crash, test_crash = train_test_split(frames_name_crash, test_size=0.2, random_state=42)

train_normal, val_normal = train_test_split(temporary_normal, test_size=0.25, random_state=42)
train_crash, val_crash = train_test_split(temporary_crash, test_size=0.25, random_state=42)


# In[6]:


print("Normal Training Videos:", train_normal)
print()
print("Crash Training Videos:", train_crash)
print()
print("Normal Validation Videos:", val_normal)
print()
print("Crash Validation Videos:", val_crash)
print()
print("Normal Test Videos:", test_normal)
print()
print("Crash Test Videos:", test_crash)


# ## Data Preprocessing

# In[7]:


# Transform image to matrix format

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=CHAN_SIZE)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0 # Normalization
    return image

train_videos = []
test_videos = []
train_labels = []
test_labels = []

for t in train_normal:
    video = []
    for i in range(50):
        current_frame_index = str(i)
        if (i < 10):
            video.append(load_image(frames_path_normal + t + "/frame_000" + str(i) + ".jpg"))
        else:
            video.append(load_image(frames_path_normal + t + "/frame_00" + str(i) + ".jpg"))
    video = tf.stack(video)
    train_videos.append(video.numpy())
    train_labels.append(0)
    
for t in test_normal:
    video = []
    for i in range(50):
        current_frame_index = str(i)
        if (i < 10):
            video.append(load_image(frames_path_normal + t + "/frame_000" + str(i) + ".jpg"))
        else:
            video.append(load_image(frames_path_normal + t + "/frame_00" + str(i) + ".jpg"))
    video = tf.stack(video)
    test_videos.append(video.numpy())
    test_labels.append(0)

for t in train_crash:
    video = []
    for i in range(50):
        current_frame_index = str(i)
        if (i < 10):
            video.append(load_image(frames_path_crash + t + "/frame_000" + str(i) + ".jpg"))
        else:
            video.append(load_image(frames_path_crash + t + "/frame_00" + str(i) + ".jpg"))
    video = tf.stack(video)
    train_videos.append(video.numpy())
    train_labels.append(1)
    
for t in test_crash:
    video = []
    for i in range(50):
        current_frame_index = str(i)
        if (i < 10):
            video.append(load_image(frames_path_crash + t + "/frame_000" + str(i) + ".jpg"))
        else:
            video.append(load_image(frames_path_crash + t + "/frame_00" + str(i) + ".jpg"))
    video = tf.stack(video)
    test_videos.append(video.numpy())
    test_labels.append(1)

train_videos = np.asarray(train_videos)
test_videos = np.asarray(test_videos)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

valid_videos = []
valid_labels = []
    
for t in val_crash:
    video = []
    for i in range(50):
        current_frame_index = str(i)
        if (i < 10):
            video.append(load_image(frames_path_crash + t + "/frame_000" + str(i) + ".jpg"))
        else:
            video.append(load_image(frames_path_crash + t + "/frame_00" + str(i) + ".jpg"))
    video = tf.stack(video)
    valid_videos.append(video.numpy())
    valid_labels.append(1)

for t in val_normal:
    video = []
    for i in range(50):
        current_frame_index = str(i)
        if (i < 10):
            video.append(load_image(frames_path_normal + t + "/frame_000" + str(i) + ".jpg"))
        else:
            video.append(load_image(frames_path_normal + t + "/frame_00" + str(i) + ".jpg"))
    video = tf.stack(video)
    valid_videos.append(video.numpy())
    valid_labels.append(0)

valid_videos = np.asarray(valid_videos)
valid_labels = np.asarray(valid_labels)


# In[8]:


# Create Dataloader

def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


trainloader = prepare_dataloader(train_videos, train_labels, "train")
testloader = prepare_dataloader(test_videos, test_labels, "test")
validloader = prepare_dataloader(valid_videos, valid_labels, "valid")

# Create Embedding Mechanism

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        
        # `projected_patches`
        # dividing the input into patches (determined by kernel_size and strides) 
        # and transforming each patch into an 64-dimensional embedding.
        
        projected_patches = self.projection(videos) 
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

# Create Positional Mechanism

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = ops.arange(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens


# ## Run Model

# In[9]:


""" REFERENCE

IMG_SIZE = 128
CHAN_SIZE = 1
BATCH_SIZE = 8
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (50, IMG_SIZE, IMG_SIZE, CHAN_SIZE)
NUM_CLASSES = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

PATCH_SIZE = (8,8,8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

LAYER_NORM_EPS = 1e-6
EMBED_DIM = 64
NUM_HEADS =  6
NUM_LAYERS = 6

"""

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=EMBED_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape) # shape=(50,128,128,1)
    # Create patches
    patches = tubelet_embedder(inputs)
    # Encode patches
    encoded_patches = positional_encoder(patches)

    for _ in range(transformer_layers):
        
        # 1. Layer normalization and MultiHeadAttention
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # 2. The MultiHeadAttention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)
        
        # 3. Skip connection - Add output from dense layers to earlier layer normalization
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # 4. Layer Normalization and MultiLayerPerception
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # 5. The two fully-connected layers with GELU activation functions
        x4 = layers.Dropout(0.1)(layers.Dense(units=embed_dim, activation='gelu')(x3))
        x5 = layers.Dropout(0.1)(layers.Dense(units=embed_dim, activation='gelu')(x4))
        
        # 6. Skip connection - Add output from dense layers to earlier layer normalization
        encoded_patches = layers.Add()([x5, x2])

    # Layer normalization and Global average pooling
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# In[10]:


def run_experiment(callbacks=None):
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=EMBED_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=EMBED_DIM)
    )

    # Compile the model with the optimizer, loss function and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Train the model
    # You may definn the epochs here
    print("Callbacks being passed to model.fit():", callbacks)
    if (callbacks != None):
        history = model.fit(trainloader, epochs=5, validation_data=validloader, callbacks=callbacks)
    else:
        history = model.fit(trainloader, epochs=5, validation_data=validloader)

    return model, history

model, history = run_experiment()


# ## Visualization

# Specific on how many feature using, on which layer, we would provide such visualizations.
# 1. Model Evaluation
#     1. Model accuracy & loss
#     2. ROC Curves and AUC 
# 2. Model Analysis
#     1. Confusion Matrix
#     2. Feature analysis
#         1. Weight Histograms
#         2. LIME explainer
#         3. Model performance on specific features
#         4. t-SNE or PCA of Layer Activations（目前做的是2个dimensions）
#         5. Layer Weight and Activation Animations(Processing)（设想的是每一个epoch都有一个heatmap of features weights,点击begin的时候heatmap会从epoch1对应的开始变动到epoch5）
# 3. Attention Head Analysis(may or may not do)

# ## 1. Model Evaluation

# ### (i) Model accuracy & loss

# In[36]:


import matplotlib.pyplot as plt
import streamlit as st

# Create a figure object to hold the subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # Adjusted figsize for better visibility

# First subplot for Accuracy
ax[0].plot(history.history['accuracy'], color='green')
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

# Second subplot for Loss
ax[1].plot(history.history['loss'], color='green')
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

# Use Streamlit's function to display the figure
st.pyplot(fig)
# plt.show()


# ### (ii) ROC Curves and AUC

# In[37]:


# Predict the probabilities for the positive class
y_pred_probs = model.predict(valid_videos)
y_score = y_pred_probs[:, 1]

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(valid_labels, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='seagreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='limegreen', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")

# Use Streamlit's st.pyplot() to render the plot
st.pyplot(fig)
# plt.show()


# ## 2. Model Analysis

# ### 2.1 Confusion matrix (TF vs. TN vs. FP vs. FN) 
# Result Evaluation  - Bias towards Classifying Videos to Normal Class

# In[38]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Evaluate 16 testing videos

model_predict_test = model.predict(testloader)

# Inspect Test Results

pred = []
normal_count = 0
misclassified_indicies = []

for i in range(16):
    crash_conf = model_predict_test[i][0]
    normal_conf = model_predict_test[i][1]
    if (crash_conf > normal_conf):
#         print("Prediction: Crash", crash_conf)
        pred.append(1)
    else:
#         print("Prediction: Normal", normal_conf)
        pred.append(0)
        normal_count += 1
#     if (i < 8):
#         print("Actual:     Crash", test_normal[i]+".mp4")
#     else:
#         print("Actual:     Normal", test_crash[abs(i-8)]+".mp4")
#     print()
# print(str(abs(len(test_labels)-normal_count)) + " videos out of 16 videos are classified as Crash")
# print(str(normal_count) + " videos out of 16 videos are classified as Normal")

true = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]

cm = confusion_matrix(true, pred)

# Visualize the confusion matrix using seaborn
fig, ax = plt.subplots()  # Create a figure and a set of subplots
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_xlabel('Predicted Labels (0-Normal, 1-Crash)')
ax.set_ylabel('True Labels (0-Normal, 1-Crash)')
ax.set_title('Confusion Matrix of Predicted vs. True Labels')

# If using Streamlit, use st.pyplot() to render matplotlib figures
st.pyplot(fig)


# ### 2.2 Feature analysis (Weight and Bias)

# In[ ]:


weights,biases = model.layers[-1].get_weights()


# In[39]:


def feature_visual(display, rank_style, num_feature):
    weights,biases = model.layers[-1].get_weights()
    bound = math.ceil(max(abs(np.max(weights)), abs(np.min(weights)))*10)/10
    
    differences = abs(weights[:, 1]) - abs(weights[:, 0])
    
    # Rank Styles:
    sorted_indices = [int(i) for i in range(EMBED_DIM)]
    if (rank_style == 'Normal (Top First)'): # rank by greatest contribution to normal class
        sorted_indices = np.argsort(weights[:, 0])[::-1]
    elif (rank_style == 'Normal (Bottom First)'): # rank by smallest contribution to normal class
        sorted_indices = np.argsort(weights[:, 0])
    elif (rank_style == 'Crash (Top First)'): # rank by greatest contribution to crash class
        sorted_indices = np.argsort(weights[:, 1])[::-1]
    elif (rank_style == 'Crash (Bottom First)'): # rank by smallest contribution to crash class
        sorted_indices = np.argsort(weights[:, 1])
    
    # Summary:
    # (1) How many feature has tendency towards classifying model to Crash/Normal Class
    negative_count = np.sum(differences < 0) # Normal
    positive_count = np.sum(differences > 0) # Crash
    
    # (2) Greatest Crash POSITIVE Influence
    max_crash_positive_influence = np.max(weights[:, 1][weights[:, 1] > 0])
    
    # (3) Smallest Crash POSITIVE Influence
    min_crash_positive_influence = np.min(weights[:, 1][weights[:, 1] > 0])
    
    # (4) Greatest Normal POSITIVE Influence
    max_normal_positive_influence = np.max(weights[:, 0][weights[:, 1] > 0])
    
    # (5) Smallest Normal POSITIVE Influence
    min_normal_positive_influence = np.min(weights[:, 0][weights[:, 1] > 0])
    
    info = f'Total Number of Features: {EMBED_DIM}\n{negative_count} Features Contribute Mainly to Normal Class\n{positive_count} Features Contribute Mainly to Crash Class\nMax Crash Positive Influence: {max_crash_positive_influence}\nMin Crash Positive Influence: {min_crash_positive_influence}\nMax Normal Positive Influence: {max_normal_positive_influence}\nMin Normal Positive Influence: {min_normal_positive_influence}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)

    
    # Plotting
    if (display == 'All'):
        index = np.arange(2)
        if (num_feature>1):
            fig, axs = plt.subplots(num_feature, 1, figsize=(5, num_feature+0.5))
            for i in range(num_feature):
                axs[i].barh(index, weights[sorted_indices[i], :], 1, color=['skyblue', 'pink']) ##
                axs[i].set_yticks(index)
                axs[i].set_yticklabels(['Normal', 'Crash'])
                axs[i].set_title('Feature {}'.format(sorted_indices[i]))
                axs[i].set_xlim([-bound, bound])
                axs[i].set_xlabel('Influence')
        else:
            fig, ax = plt.subplots(figsize=(5, 1.26))
            ax.barh(index, weights[sorted_indices[0], :], 1, color=['skyblue', 'pink'])
            ax.set_yticks(index)
            ax.set_yticklabels(['Normal', 'Crash'])
            ax.set_title('Feature {}'.format(sorted_indices[0]))
            ax.set_xlim([-bound, bound])
            ax.set_xlabel('Influence')
    else:
        if (num_feature>1):
            fig, axs = plt.subplots(num_feature, 1, figsize=(5, num_feature))
            for i in range(num_feature):
                axs[i].barh(0, differences[sorted_indices[i]], color='pink' \
                            if differences[sorted_indices[i]] >= 0 else 'skyblue')
                axs[i].set_title('Feature {}'.format(sorted_indices[i]))
                axs[i].set_xlim([-bound, bound])
                axs[i].set_xlabel('Influence Tendency')
                axs[i].axes.get_yaxis().set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=(5, 1.1))
            ax.barh(0, differences[sorted_indices[0]], color='pink' \
                    if differences[sorted_indices[0]] >= 0 else 'skyblue')
            ax.set_title('Feature {}'.format(sorted_indices[0]))
            ax.set_xlim([-bound, bound])
            ax.set_xlabel('Influence Tendency')
            ax.axes.get_yaxis().set_visible(False)
    fig.text(1.05, 1, info, transform=fig.transFigure, fontsize=12, verticalalignment='top', bbox=props)
#     plt.tight_layout()
    st.pyplot(fig)
    


# In[ ]:


# Convert ipywidgets UI into Streamlit UI
display_options = ['All', 'Tendency']
rank_style_options = [
    'Default',
    'Normal (Top First)', 
    'Normal (Bottom First)', 
    'Crash (Top First)',
    'Crash (Bottom First)'
]

# Create Streamlit UI
display = st.selectbox('Display:', display_options)
rank_style = st.selectbox('Rank Style:', rank_style_options)
num_feature = st.slider('Number of Features:', 1, EMBED_DIM-1, value=3)

# Call function with Streamlit widgets' values
feature_visual(display, rank_style, num_feature)


# In[ ]:




