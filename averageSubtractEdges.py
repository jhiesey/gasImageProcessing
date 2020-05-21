#!/usr/bin/env python3

import numpy
import cv2
import sys

infile = sys.argv[1]
outfile = None
if len(sys.argv) > 2:
	outfile = sys.argv[2]

SATURATION = 8
OVERLAY_HUE = 0 # red
EDGES_CONTRAST = 0.3

EDGES_THRESH1 = 40
EDGES_THRESH2 = 80

frames = []
sumFrame = None
cap = cv2.VideoCapture(infile)
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
	ret, rawFrame = cap.read()
	if ret:
		frameInt = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2GRAY)
		frameFloat = frameInt.astype(numpy.float32)
		frames.append(frameFloat)

		if sumFrame is None:
			sumFrame = frameFloat.copy()
		else:
			sumFrame += frameFloat
	else:
		break
cap.release()

averageFrame = sumFrame / len(frames)

hsv = numpy.zeros((averageFrame.shape[0], averageFrame.shape[1], 3), numpy.uint8)
hsv[...,0] = OVERLAY_HUE

writer = None
if outfile is not None:
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	writer = cv2.VideoWriter(outfile, fourcc, fps, (averageFrame.shape[1], averageFrame.shape[0]), True)

for frame in frames:
	differenceFrame = numpy.clip((averageFrame - frame) * SATURATION, 0, 255).astype(numpy.uint8)
	edges = (255 - cv2.Canny(frame.astype(numpy.uint8), EDGES_THRESH1, EDGES_THRESH2) * EDGES_CONTRAST)

	hsv[...,1] = differenceFrame
	hsv[...,2] = edges

	rgbFrame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	if writer is not None:
		writer.write(rgbFrame)
	else:
		cv2.imshow('Average subtracted', rgbFrame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

if writer is not None:
	writer.release()

cv2.destroyAllWindows()

