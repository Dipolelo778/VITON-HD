# network.py

from segmentation import SegmentationNet
from cloth_parser import ClothParser
from gmm import GMM
from refiner import RefinerUNet
from inpainting import Inpainter

class VITONHD:
    def __init__(self):
        self.segmentation = SegmentationNet()
        self.cloth_parser = ClothParser()
        self.gmm = GMM()
        self.refiner = RefinerUNet()
        self.inpainter = Inpainter()
                      


