import sys
import cv2
import os
import argparse
import time

def setup_openpose()
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('./openpose/build/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

    params = dict()
    params["model_folder"] = "./openpose/models/"
    params["model_pose"] = "COCO"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return opWrapper

def openpose(image, opWrapper):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])

    return datum.poseKeypoints
