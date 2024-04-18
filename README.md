# Car Accident Classifier

## Data Source
**GitHub Page**: https://github.com/Cogito2012/CarCrashDataset
**Google Drive**: https://drive.google.com/drive/folders/1Rx4LCo-9AbAdPw5Zh7wpKhMKLabZ5oA8

## Directory Tree
```
.
├── Split.ipynb  # Code for splitting videos into frames
├── Crash-1500.txt  # info data of 1500 car crash videos
├── 3001 Visualization final project.ipynb # Code for extracting video frames and corresponding feature maps in CNN
└── README.md
```

## Plan for now
Transformer
<ol>
  <li>OpenCV/FFmpeg for dividing videos into frames - treat each frame as an element in the sequence</li>
</ol>
1) 
2) Use a CNN to extract features from each frame (video-frame data are still very high dimensional)
3) Feed the sequences of frames into Video Transformer
4) Modifications of the attention mechanisms
5) Design the complexity of the videos (depth, width, attention)
