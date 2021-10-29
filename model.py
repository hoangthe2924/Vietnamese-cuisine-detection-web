import os
import cv2

cwd = os.path.dirname(os.path.realpath(__file__)) + "\\model\\"

class Model:
    def __init__(self, file_label, file_weights, file_cfg):
        labelsPath = cwd + file_label
        weightsPath = cwd + file_weights
        configPath = cwd + file_cfg
        self.classes = open(labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)