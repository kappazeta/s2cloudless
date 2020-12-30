from s2cloudless import S2PixelCloudDetector
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
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
    save_to=inpu_folder.replace("IMG_DATA","S2CLOUDLESS_DATA")
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
        im.result.save(os.path.join(save_to,"s2cloudless_prediction.png"))

def plot_probability_map(prob_map, figsize=(15, 15)):
    if(a.mode=="validation"):
        plt.figure(figsize=figsize)
        plt.imshow(prob_map,cmap=plt.cm.inferno)
        plt.savefig(identifier+"_s2cloudless_probability.png")
    else:
        im_result = Image.fromarray(np.uint8(prob_map))
        im_result.save(os.path.join(save_to,"s2cloudless_probability.png"))       

#Read in the bands and resample by B01 (60 m)

with rasterio.open(os.path.join(input_folder,identifier+"B01.jp2")) as scl:
    B01=scl.read()
    tmparr = np.empty_like(B01)
    aff = scl.transform
    print(B01.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B02.jp2")) as scl:
    B02=scl.read()    
    reproject(B02, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B02 = tmparr
    print(B02.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B04.jp2")) as scl:
    B04=scl.read()
    reproject(B04, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B04 = tmparr
    print(B04.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B05.jp2")) as scl:
    B05=scl.read()
    reproject(B05, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B05 = tmparr
    print(B05.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B08.jp2")) as scl:
    B08=scl.read()
    reproject(B08, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B08 = tmparr
    print(B08.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B8A.jp2")) as scl:
    B8A=scl.read()
    reproject(B8A, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B8A = tmparr
    print(B8A.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B09.jp2")) as scl:
    B09=scl.read()
    print(B09.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B10.jp2")) as scl:
    B10=scl.read()
    print(B10.shape)  
    
with rasterio.open(os.path.join(input_folder,identifier+"B11.jp2")) as scl:
    B11=scl.read()
    reproject(B11, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B11 = tmparr
    print(B11.shape)
    
with rasterio.open(os.path.join(input_folder,identifier+"B12.jp2")) as scl:
    B12=scl.read()
    reproject(B12, tmparr,src_transform = scl.transform,dst_transform = aff,src_crs = scl.crs,dst_crs = scl.crs,resampling = Resampling.bilinear)
    B12 = tmparr
    print(B12.shape)


bands = np.array([np.dstack((B01[0]/10000.0,B02[0]/10000.0,B04[0]/10000.0,B05[0]/10000.0,B08[0]/10000.0,B8A[0]/10000.0,B09[0]/10000.0,B10[0]/10000.0,B11[0]/10000.0,B12[0]/10000.0))])

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)  #These are the recommended parameters for this resolution (60m)
cloud_probs = cloud_detector.get_cloud_probability_maps(bands)
mask = cloud_detector.get_cloud_masks(bands).astype(rasterio.uint8)

plot_cloud_mask(mask[0])
plot_probability_map(cloud_probs[0])




