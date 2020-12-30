# s2cloudless

S2cloudless is Sentinel Hub's cloud detector for Sentinel-2 imagery. It is a binary classifier (cloud and non-cloud). The output is prediction mask (binary) and probability map. The resolution of the output is 60m. S2cloudless was trained on L1C data it cannot perform well on L2A data.

The script run_s2cloudless will run this cloud detector on a local L1C product.

Two arguments are needed to run the script:

--input - path to IMG_DATA folder, where the jp2 files of the L1C product are located
--mode - either "validation" or "CVAT-VSM"

Validation

-Output images rescaled to 10980 x 10980.
-Mask output with pixel values 0 or 255.
-Probability output with colormap.
-The result will be saved in the same directory where the script is being run

CVAT-VSM

-Output images in native dimensions 1830 x 1830.
-Mask output with pixel values 0 or 1.
-Probability output as grayscale (also, not normalized).
-The result will be saved to a specified folder
