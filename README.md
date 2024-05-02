# Car Accident Classifier

## Data Source

**GitHub Page**: https://github.com/Cogito2012/CarCrashDataset

**Google Drive**: https://drive.google.com/drive/folders/1Rx4LCo-9AbAdPw5Zh7wpKhMKLabZ5oA8

## Directory Tree
```
.
├── Split.ipynb                            # Code for splitting videos into frames
├── VideoVisionTransformer.ipynb           # ViVT model for video prediction
├── 3001 Visualization final project.ipynb # Code for extracting video frames and corresponding feature maps in CNN
└── README.md
```

## Plan for now
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

It should also describe your design goals for your visualizations (what types of insights you hope the user will be able to get from your design)