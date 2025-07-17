# network.py

from Segmentation import SegmentationNet
from Cloth_parser import ClothParser
from Gmm import GMM
from Refiner import RefinerUNet
from Inpainting import Inpainter

class VITONHD:
    def __init__(self):
        self.segmentation = SegmentationNet()
        self.cloth_parser = ClothParser()
        self.gmm = GMM()
        self.refiner = RefinerUNet()
        self.inpainter = Inpainter()
                      


