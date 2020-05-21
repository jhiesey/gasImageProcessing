#!/usr/bin/env python3

import numpy
import cv2
import sys

infile = sys.argv[1]
outfile = None
if len(sys.argv) > 2:
	outfile = sys.argv[2]

CONTRAST = 2

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

writer = None
if outfile is not None:
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	writer = cv2.VideoWriter(outfile, fourcc, fps, (averageFrame.shape[1], averageFrame.shape[0]), True)

for frame in frames:
	correctedFrame = numpy.clip((frame - averageFrame) * CONTRAST + 128, 0, 255)

	intFrame = correctedFrame.astype(numpy.uint8)

	rgbFrame = cv2.cvtColor(intFrame, cv2.COLOR_GRAY2BGR)
	if writer is not None:
		writer.write(rgbFrame)
	else:
		cv2.imshow('Average subtracted', rgbFrame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

if writer is not None:
	writer.release()

cv2.destroyAllWindows()

