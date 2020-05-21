#!/usr/bin/env python3

import numpy
import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])

OVERLAY_HUE = 0 # red

prev = None
hsv = None
while cap.isOpened():
	ret, rawFrame = cap.read()
	if ret:
		frame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2GRAY)
		if hsv is None:
			hsv = numpy.zeros_like(rawFrame)
			hsv[...,0] = OVERLAY_HUE

		if prev is not None:
			flow = cv2.calcOpticalFlowFarneback(prev, frame, None, 0.5, 10, 5, 5, 5, 1.1, 0)
			mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

			hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
			hsv[...,2] = frame

			flowBgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
			cv2.imshow('Flow', flowBgr)
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		prev = frame

	else:
		break

cap.release()

cv2.destroyAllWindows()

