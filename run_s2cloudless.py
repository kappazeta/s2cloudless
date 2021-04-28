from s2cloudless import S2PixelCloudDetector
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="path to the folder where the jp2 files of the L1C product are located (IMG_DATA)")
parser.add_argument("--mode", required=True,  choices=["validation", "CVAT-VSM"], help="validation: output images rescaled to 10980x10980px, mask output with pixel values 0 or 255, probability output with colormap. CVAT-VSM: output image dimensions 1830x1830px, mask output with pixel values 0 or 1, probabilty output as greyscaled.")
a = parser.parse_args()

input_folder=a.input
save_to=""

if(a.mode=="CVAT-VSM"):
    save_to=input_folder.replace("IMG_DATA","S2CLOUDLESS_DATA")
    if(os.path.isdir(save_to)==False):
        os.makedirs(save_to)
        print("Created folder "+save_to)


#Check the name of the product in the folder

identifier=""
for filename in os.listdir(input_folder):
    if("B01.jp2" in filename):
        identifier=filename.split("B01")[0]
        
def plot_cloud_mask(mask, figsize=(15, 15), fig=None):
    """
    Utility function for plotting a binary cloud mask.
    """ 
    if(a.mode=="validation"): 
        new_mask = [[255 if b==1 else b for b in i] for i in mask]
        im_result = Image.fromarray(np.uint8(new_mask))
        im_result=im_result.resize((10980,10980),Image.NEAREST)
        im_result.save(identifier+"_s2cloudless_prediction.png")
    else:
        im_result = Image.fromarray(np.uint8(mask))
        im_result.save(os.path.join(save_to,"s2cloudless_prediction.png"))

def plot_probability_map(prob_map, figsize=(15, 15)):
    if(a.mode=="validation"):
        plt.figure(figsize=figsize)
        plt.imshow(prob_map,cmap=plt.cm.inferno)
        plt.savefig(identifier+"_s2cloudless_probability.png")
    else:
        im_result = Image.fromarray(np.uint8(prob_map * 255))
        im_result.save(os.path.join(save_to,"s2cloudless_probability.png"))       

#Resampling to achieve 10 m resolution:

with rasterio.open(os.path.join(input_folder,identifier+"B01.jp2")) as dataset:
    B01 = dataset.read(out_shape=(dataset.count,int(dataset.height * 6),int(dataset.width * 6)),resampling=Resampling.bilinear)                  
    print(B01.shape)

with rasterio.open(os.path.join(input_folder,identifier+"B02.jp2")) as dataset:
    B02 = dataset.read(out_shape=(dataset.count,int(dataset.height * 1),int(dataset.width * 1)),resampling=Resampling.bilinear)
    print(B02.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B04.jp2")) as dataset:
    B04 = dataset.read(out_shape=(dataset.count,int(dataset.height * 1),int(dataset.width * 1)),resampling=Resampling.bilinear)
    print(B04.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B05.jp2")) as dataset:
    B05 = dataset.read(out_shape=(dataset.count,int(dataset.height * 2),int(dataset.width * 2)),resampling=Resampling.bilinear)
    print(B05.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B08.jp2")) as dataset:
    B08 = dataset.read(out_shape=(dataset.count,int(dataset.height * 1),int(dataset.width * 1)),resampling=Resampling.bilinear)
    print(B08.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B8A.jp2")) as dataset:
    B8A = dataset.read(out_shape=(dataset.count,int(dataset.height * 2),int(dataset.width * 2)),resampling=Resampling.bilinear)
    print(B8A.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B09.jp2")) as dataset:
    B09 = dataset.read(out_shape=(dataset.count,int(dataset.height * 6),int(dataset.width * 6)),resampling=Resampling.bilinear)
    print(B09.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B10.jp2")) as dataset:
    B10 = dataset.read(out_shape=(dataset.count,int(dataset.height * 6),int(dataset.width * 6)),resampling=Resampling.bilinear)
    print(B10.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B11.jp2")) as dataset:
    B11 = dataset.read(out_shape=(dataset.count,int(dataset.height * 2),int(dataset.width * 2)),resampling=Resampling.bilinear)
    print(B11.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B12.jp2")) as dataset:
    B12 = dataset.read(out_shape=(dataset.count,int(dataset.height * 2),int(dataset.width * 2)),resampling=Resampling.bilinear)
    print(B12.shape)

bands = np.array([np.dstack((B01[0]/10000.0,B02[0]/10000.0,B04[0]/10000.0,B05[0]/10000.0,B08[0]/10000.0,B8A[0]/10000.0,B09[0]/10000.0,B10[0]/10000.0,B11[0]/10000.0,B12[0]/10000.0))])

#Recommended parameters for 60 m resolution: average_over = 4, dilation_size=2
#Recommended parameters for 10 m resolution: average_over=22, dilation_size=11
#The actual best result is achievable by trying different values for different products.

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=22, dilation_size=11)  
cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
mask = cloud_detector.get_cloud_masks(bands).astype(rasterio.uint8)

plot_cloud_mask(mask[0])
plot_probability_map(cloud_probs[0])
