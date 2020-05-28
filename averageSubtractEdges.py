#!/usr/bin/env python3

import numpy as np
import cv2
import sys

FILTER_K = 2

OVERLAY_SATURATION = 8
OVERLAY_BRIGTNESS = 0.5
OVERLAY_NEGATIVE_HUE = 0 # red
OVERLAY_POSITIVE_HUE = 120 # blue

EDGES_CONTRAST = 0.3
EDGES_THRESH1 = 40
EDGES_THRESH2 = 80

frames = []
infile = sys.argv[1]
cap = cv2.VideoCapture(infile)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if len(sys.argv) > 2:
	outfile = sys.argv[2]
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height), True)

low_pass_frame = None
while cap.isOpened():
	ret, raw_frame = cap.read()
	if not ret:
		break

	# Convert input to black and white
	frame_int = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
	frame_float = frame_int.astype(np.float32)
	if low_pass_frame is None:
		low_pass_frame = frame_float

	# Compute change
	difference_frame = (frame_float - low_pass_frame) * OVERLAY_SATURATION
	difference_frame_abs = np.clip(np.absolute(difference_frame), 0, 255).astype(np.uint8)

	# Compute edges
	edges_frame = cv2.Canny(frame_int, EDGES_THRESH1, EDGES_THRESH2)

	# Combine edges with a lightened version of the original image to make a background reference image
	reference_channel = cv2.addWeighted(edges_frame, -EDGES_CONTRAST, frame_int, 1 - OVERLAY_BRIGTNESS, OVERLAY_BRIGTNESS * 255)

	# Set up an HSV output image to combine background as brightness and difference as color saturation
	hsv_frame = np.zeros((height, width, 3), np.uint8)

	# Determine overlay hue depending on sign of difference
	hsv_frame[...,0] = np.where(difference_frame >= 0, OVERLAY_POSITIVE_HUE, OVERLAY_NEGATIVE_HUE)

	# Difference determines color saturation/intensity
	hsv_frame[...,1] = difference_frame_abs

	# Reference determines value/brightness
	hsv_frame[...,2] = reference_channel

	# Display output
	processed_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
	if writer is not None:
		writer.write(processed_frame)
	else:
		cv2.imshow('Processed', processed_frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	# Compute average frame
	low_pass_frame += (FILTER_K / 10) * (frame_float - low_pass_frame)

cap.release()

if writer is not None:
	writer.release()

cv2.destroyAllWindows()
