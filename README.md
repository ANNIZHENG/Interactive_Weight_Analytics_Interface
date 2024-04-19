# Car Accident Classifier

## Data Source

**GitHub Page**: https://github.com/Cogito2012/CarCrashDataset

**Google Drive**: https://drive.google.com/drive/folders/1Rx4LCo-9AbAdPw5Zh7wpKhMKLabZ5oA8

## Directory Tree
```
.
├── Crash-1500.txt  # info data of 1500 car crash videos
├── Split.ipynb  # Code for splitting videos into frames
├── VideoVisionTransformer.ipynb  # ViVT model for video prediction
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
