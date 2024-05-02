# Car Accident Classifier

## Data Source

**GitHub Page**: https://github.com/Cogito2012/CarCrashDataset

**Google Drive**: https://drive.google.com/drive/folders/1Rx4LCo-9AbAdPw5Zh7wpKhMKLabZ5oA8

## Directory Tree
```
.
├── Split.ipynb                  # Code for splitting videos into frames
├── VideoVisionTransformer.ipynb # ViVT model and demo of interactive weight visualization interface
├── int_weight_vis.py            # Interactive weight visualization interface package
├── InteractiveWeightAnalysisPre # Presentation video of our interface (Demo included)
└── README.md
```

## Related Work: SHAP

## Background:

**Model Training**

1. Dataset comprises 4,500 videos (3,000 normal driving videos and 1,500 crash-related videos).
2. We utilized only 50 normal and 50 crash videos from the dataset.
3. We used Video Vision Transformer for classification.

**Design Goal**

Providing an easily accessible visualization tool for analyzing a large amount of feature weights.

## Methods

**Model Weight Retrieval**

1. Tensorflow weight retrieval
2. Tensorflow Callback Mechanism

**Interface**

1. Matplotlib
2. Jupyter Notebook Widget

**Filter Definition**

1. Sort the retrieved weights
2. Display the first x weights

**Tendency Definition**

1. Take the absolute value of both influences
2. Compare which influence is larger

## Evaluation

**Insight 1**: Some features contribute to both classes

**Insight 2**: This feature has one of the most positive influence to classification of Normal class. It, at the same time, contributes to the classification of Crash class more.

**Insight 3**: Feature weights change across epochs

## Limitation

1. Can only be displayed in Jupyter Notebook environment
2. Only binary classification tensorflow model can use it

<!-- ## Plan for now
Transformer
<ol>
  <li>Done: OpenCV/FFmpeg for dividing videos into frames - treat each frame as an element in the sequence</li>
  <li>Done: Use a CNN to extract features from each frame (video-frame data are still very high dimensional)</li>
  <li>Feed the sequences of frames into Video Transformer</li>
  <li>Modifications of the attention mechanisms</li>
  <li>Design the complexity of the videos (depth, width, attention)</li>
</ol>
**Introduction**
Why is it important? 
- Customizable (sorting mechanism)
- Inspired by analytics tools like SHAP, we provide an analytics tools for demonstrating feature importance
- Interactive tools for sorting 
- Interactive tools for analyzing weights over epochs in video vision transformer model.
What challenges have inhibited previous works from solving it? 
How did you address this problem through data visualization (e.g. "Our main contributions will be...")? 
**Related Works**
Situate your research question within the field of data visualization/machine learning. How have previous works addressed this problem, or ones similar to it? 
How have these previous works fallen short (i.e. what gaps exist in related research)?
**Background**
This section should describe your dataset.
It should also describe your design goals for your visualizations (what types of insights you hope the user will be able to get from your design) -->