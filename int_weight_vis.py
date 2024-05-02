import math
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.manifold import (TSNE)
from sklearn.metrics import roc_curve, auc
import ipywidgets as widgets
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
import seaborn as sns
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True) 

# param: tf.keras.callbacks.Callback
class WeightsHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Use `logs=None` as a default instead of `logs={}` to follow the common practice.
        self.weights_over_time = []
    def on_epoch_end(self, epoch, logs=None):
        weights, biases = self.model.layers[-1].get_weights()
        self.weights_over_time.append(weights.copy())  # Use `weights.copy()` to ensure a true copy is saved

# param: model: 
# param: optimizer:
# param: loss: 
# param: epochs: 
# param: train_dataloader: 
# param: test_dataloader: 
# param: validation_dataloader (optional): 
# param: class_names (optional): 
def run(model, optimizer, loss, epochs, train_dataloader, test_dataloader, validation_dataloader=None, class_names=[]):
    # Compile Model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    weights_history_callback = WeightsHistory()

    # Train Model
    if (validation_dataloader):
        history = model.fit(x=train_dataloader, epochs=epochs, validation_data=validation_dataloader, callbacks=[weights_history_callback])
    else:
        history = model.fit(x=train_dataloader, epochs=epochs, callbacks=[weights_history_callback])

    print("Finished training...")

    weights,biases = model.layers[-1].get_weights()

    test_data = []
    test_labels= []

    # Iterate through the dataloader
    try:
        for videos, labels in test_dataloader:
            test_data.append(videos.numpy())
            test_labels.append(labels.numpy())
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
    except Exception:
        print("Bilding visualizations...")

    # test_data = np.concatenate(test_data, axis=0)
    # test_labels = np.concatenate(test_labels, axis=0)

    model_predict_test = model.predict(test_dataloader)
    weights_over_time = weights_history_callback.weights_over_time
    
    # ## Weight Distribution
    # plt.figure(figsize=(12, 3))
    # plt.subplot(1, 2, 1)
    # plt.title("Weight Distribution of the Last Layer")
    # plt.hist(weights.flatten(), bins=50, color="black")
    # plt.xlabel("Weight values")
    # plt.ylabel("Frequency")
    
    # ## Bias Distribution
    # plt.title("Bias Distribution from the Last Layer")
    # plt.subplot(1, 2, 2)
    # plt.hist(biases.flatten(), bins=50, color="black")
    # plt.xlabel("Bias values")
    # plt.ylabel("Frequency")
    # plt.tight_layout()
    # plt.show()

    ## T-SNE Graph on Features
    # param: perplexity
    def t_sne_visual(perplexity):
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300, learning_rate=200)
        X_tsne = tsne.fit_transform(model_predict_test)
        plt.figure(figsize=(4, 3))
        unique_labels = np.unique(test_labels)
        colors = ['skyblue', 'pink']
        for i, label in enumerate(unique_labels):
            plt.scatter(X_tsne[test_labels == label, 0],
                        X_tsne[test_labels == label, 1], 
                        color=colors[i], 
                        label=f'Cluster {label}')
        plt.title('T-SNE')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

    # Options
    per_slider = widgets.IntSlider(value=10, min=1, max=len(test_labels), step=1, description='Perplexity:', continuous_update=False)
    ui1 = widgets.HBox([per_slider])
    out1 = widgets.interactive_output(t_sne_visual, {'perplexity': per_slider})
    # UI
    display(ui1, out1)

    # param: display
    # param: rank_style
    # param: num_feature
    def feature_visual(display, rank_style, num_feature):

        """
        Function that takes the parameters of
        How users from to display the weights of the features
        and how users want to rank those features
        and how many numbers of features users want to see
        """

        bound = math.ceil(max(abs(np.max(weights)), abs(np.min(weights)))*10)/10
        
        differences = abs(weights[:, 1]) - abs(weights[:, 0])
        
        # Rank Styles:
        sorted_indices = [int(i) for i in range(len(weights))]

        if (len(class_names) == 2):
            if (rank_style == str(class_names[0])+' (DESC)'):
                sorted_indices = np.argsort(weights[:, 0])[::-1]
            elif (rank_style == str(class_names[0])+' (ASC)'):
                sorted_indices = np.argsort(weights[:, 0])
            elif (rank_style == str(class_names[1])+' (DESC)'):
                sorted_indices = np.argsort(weights[:, 1])[::-1]
            elif (rank_style == str(class_names[1])+' (ASC)'):
                sorted_indices = np.argsort(weights[:, 1])
        else:
            if (rank_style == 'Class 0 (DESC)'): 
                sorted_indices = np.argsort(weights[:, 0])[::-1]
            elif (rank_style == 'Class 0 (ASC)'): 
                sorted_indices = np.argsort(weights[:, 0])
            elif (rank_style == 'Class 1 (DESC)'): 
                sorted_indices = np.argsort(weights[:, 1])[::-1]
            elif (rank_style == 'Class 1 (ASC)'): 
                sorted_indices = np.argsort(weights[:, 1])
        
        # Summary:
        # (1) How many feature has tendency towards classifying model to Class 0 / Class 1 Class
        negative_count = np.sum(differences < 0) # Class 1
        positive_count = np.sum(differences > 0) # Class 0
        
        # (2) Greatest Class 0 POSITIVE Influence
        max_class1_positive_influence = np.max(weights[:, 1][weights[:, 1] > 0])
        
        # (3) Smallest Class 0 POSITIVE Influence
        min_class1_positive_influence = np.min(weights[:, 1][weights[:, 1] > 0])
        
        # (4) Greatest Class 1 POSITIVE Influence
        max_class0_positive_influence = np.max(weights[:, 0][weights[:, 1] > 0])
        
        # (5) Smallest Class 1 POSITIVE Influence
        min_class0_positive_influence = np.min(weights[:, 0][weights[:, 1] > 0])
        
        info = f'Total Number of Features: {len(weights)}\n{negative_count} Features Contribute Mainly to Normal Class\n{positive_count} Features Contribute Mainly to Crash Class\nMax Crash Positive Influence: {max_class0_positive_influence}\nMin Crash Positive Influence: {min_class0_positive_influence}\nMax Normal Positive Influence: {max_class1_positive_influence}\nMin Normal Positive Influence: {min_class1_positive_influence}'

        # Plotting
        if (display == 'All'):
            index = np.arange(2)
            if (num_feature>1):
                fig, axs = plt.subplots(num_feature, 1, figsize=(5, num_feature+0.5))
                for i in range(num_feature):
                    axs[i].barh(index, weights[sorted_indices[i], :], 1, color=['skyblue', 'pink'])
                    axs[i].set_yticks(index)
                    if (len(class_names) == 2):
                        axs[i].set_yticklabels(class_names)
                    else:
                        axs[i].set_yticklabels(['Class 0', 'Class 1'])
                    axs[i].set_title('Feature {}'.format(sorted_indices[i]))
                    axs[i].set_xlim([-bound, bound])
                    axs[i].set_xlabel('Influence')
            else:
                fig, ax = plt.subplots(figsize=(5, 1.26))
                ax.barh(index, weights[sorted_indices[0], :], 1, color=['skyblue', 'pink'])
                ax.set_yticks(index)
                if (len(class_names) == 2):
                    ax.set_yticklabels(class_names)
                else:
                    ax.set_yticklabels(['Class 0', 'Class 1'])
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
        
        legend_patches = [Patch(color='pink', label='Class 1'), Patch(color='skyblue', label='Class 0')]
        if (len(class_names) == 2):
            legend_patches = [Patch(color='pink', label=class_names[1]),Patch(color='skyblue', label=class_names[0])]

        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.31, 1), borderaxespad=0.)
        print(info)
        plt.tight_layout()
        plt.show()

    # Options
    display_widget = widgets.Dropdown(options=['All', 'Tendency'], description='Display:')
    rankstyle_widget = widgets.Dropdown(options=['Default','Class 1 (DESC)', 'Class 1 (ASC)', 'Class 0 (DESC)','Class 0 (ASC)'], description='Rank:') ###
    if (len(class_names) == 2):
        # Normal, Crash
        rankstyle_widget = widgets.Dropdown(options=['Default', class_names[0]+' (DESC)', class_names[0]+' (ASC)', class_names[1]+' (DESC)', class_names[1]+' (ASC)'], description='Rank:')
    num_feature_widget = widgets.IntSlider(value=3, min=1, max=(len(weights)-1), step=1, description='Num Feature:')
    # UI
    ui2 = widgets.HBox([display_widget,rankstyle_widget,num_feature_widget])
    out2 = widgets.interactive_output(feature_visual, {'display': display_widget, 
                                                      'rank_style': rankstyle_widget,
                                                      'num_feature': num_feature_widget})
    display(ui2, out2)


    # param: epoch
    # param: num_feature
    # param: rank_style
    def weights_over_time_visual(epoch, num_features, rank_style):
        # Use epoch as index
        epoch = epoch-1
        
        weights = weights_over_time[epoch]
        
        sorted_indices = np.arange(weights.shape[0])
        if (len(class_names) == 2):
            if (rank_style == str(class_names[0])+' (DESC)'):
                sorted_indices = np.argsort(weights[:, 0])[::-1]
            elif (rank_style == str(class_names[0])+' (ASC)'):
                sorted_indices = np.argsort(weights[:, 0])
            elif (rank_style == str(class_names[1])+' (DESC)'):
                sorted_indices = np.argsort(weights[:, 1])[::-1]
            elif (rank_style == str(class_names[1])+' (ASC)'):
                sorted_indices = np.argsort(weights[:, 1])
        else:
            if rank_style == 'Class 0 (DESC)': 
                sorted_indices = np.argsort(weights[:, 0])[::-1]
            elif rank_style == 'Class 0 (ASC)': 
                sorted_indices = np.argsort(weights[:, 0])
            elif rank_style == 'Class 1 (DESC)': 
                sorted_indices = np.argsort(weights[:, 1])[::-1]
            elif rank_style == 'Class 1 (ASC)': 
                sorted_indices = np.argsort(weights[:, 1])

        ## Accuracy and Loss Over Time
        if (epoch>0):
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'][:(epoch+1)], color="black")
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, epoch+1, 1))
            plt.legend(['Train', 'Validation'], loc='upper left')

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'][:(epoch+1)], color="black")
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xticks(np.arange(0, epoch+1, 1))
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        else:
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.bar(x=[''], height=[history.history['accuracy'][epoch]], width=0.1, color="black")
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')

            plt.subplot(1, 2, 2)
            plt.bar(x=[''], height=[history.history['loss'][epoch]], width=0.1, color="black")
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()

        ## Weight Change Over Time

        # Apply sorting and limit to the number of features selected
        weights_to_plot = weights[sorted_indices][:num_features, :]
        feature_labels = sorted_indices[:num_features]  # Label y-axis with the feature indices

        # Plotting the heatmap
        plt.figure(figsize=(10, 20))
        if (len(class_names)==2):
            sns.heatmap(weights_to_plot, annot=True, fmt=".2f", cmap=plt.get_cmap('RdBu'),cbar=True, xticklabels=class_names, yticklabels=feature_labels)
        else:
            sns.heatmap(weights_to_plot, annot=True, fmt=".2f", cmap=plt.get_cmap('RdBu'),cbar=True, xticklabels=["Class 0", "Class 1"], yticklabels=feature_labels)
        plt.title(f"Heatmap of Weights for Epoch {epoch + 1}")
        plt.xlabel("Class Contribution")
        plt.ylabel("Feature Index")
        plt.show()

        # Statistical information based on the ranking and weights
        differences = np.abs(weights[:, 1]) - np.abs(weights[:, 0])
        negative_count = np.sum(differences < 0) # Class 0
        positive_count = np.sum(differences > 0) # Class 1

    # Options
    epoch_slider = widgets.IntSlider(value=1, min=1, max=len(weights_over_time), step=1, description='Epoch:',continuous_update=False)
    feature_slider = widgets.IntSlider(value=64, min=1, max=64, step=1, description='Features:', continuous_update=False)
    rank_style_dropdown = widgets.Dropdown(options=['Default', 'Class 0 (DESC)', 'Class 0 (ASC)', 'Class 1 (DESC)', 'Class 1 (ASC)'], value='Default', description='Rank Style:',)
    if (len(class_names) == 2):
        rank_style_dropdown = widgets.Dropdown(options=['Default', class_names[0]+' (DESC)', class_names[0]+' (ASC)', class_names[1]+' (DESC)', class_names[1]+' (ASC)'], description='Rank Style:',)

    ui3 = widgets.HBox([epoch_slider,feature_slider,rank_style_dropdown])
    out3 = widgets.interactive_output(weights_over_time_visual, {'epoch': epoch_slider, 
                                                              'num_features': feature_slider,
                                                              'rank_style': rank_style_dropdown})
    # UI
    display(ui3, out3)