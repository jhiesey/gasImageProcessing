# gasImageProcessing

Install `opencv-python` with pip

To subtract the average image and boost constrast and display the resulting video, run
```
./averageSubtract.py infile.mp4
```

To write the output video to a file, run
```
./averageSubtract.py infile.mp4 outfile.mp4
```

averageSubtractEdges.py is the same, except combines the edges of the original frames with color to show darker areas where IR is blocked.

opticalFlow.py is an experiment with optical flow, but it doesn't work very well. The output is too noisy.