# # USAGE
# # python example.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg
#
# # import the necessary packages
# from color_transfer import color_transfer
# import numpy as np
# import argparse
# import cv2
#
# import matplotlib.pyplot as plt
#
# def show_image(title, image, width = 300):
# 	# resize the image to have a constant width, just to
# 	# make displaying the images take up less screen real
# 	# estate
# 	r = width / float(image.shape[1])
# 	dim = (width, int(image.shape[0] * r))
# 	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#
# 	# show the resized image
# 	cv2.imshow(title, resized)
#
# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--source", required = True,
# 	help = "Path to the source image")
# ap.add_argument("-t", "--target", required = True,
# 	help = "Path to the target image")
# ap.add_argument("-co","--concat",required = True,
# 				help = "Path to the concat image")
# ap.add_argument("-c", "--clip", type = str2bool, default = 't',
# 	help = "Should np.clip scale L*a*b* values before final conversion to BGR? "
# 		   "Approptiate min-max scaling used if False.")
# ap.add_argument("-p", "--preservePaper", type = str2bool, default = 't',
# 	help = "Should color transfer strictly follow methodology layed out in original paper?")
# ap.add_argument("-o", "--output", help = "Path to the output image (optional)")
# args = vars(ap.parse_args())
#
# # load the images
# source = cv2.imread(args["source"])
# target = cv2.imread(args["target"])
#
# # transfer the color distribution from the source image
# # to the target image
# transfer = color_transfer(source, target, clip=args["clip"], preserve_paper=args["preservePaper"])
#
# # check to see if the output image should be saved
# if args["output"] is not None:
# 	cv2.imwrite(args["output"], transfer)
#
# # show the images and wait for a key press
# show_image("Source", source)
# show_image("Target", target)
# show_image("Transfer", transfer)
#
#
# cv2.waitKey(0)


from color_transfer import color_transfer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

def show_image(title, image, width=300):
    # Resize the image to have a constant width
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Show the resized image
    cv2.imshow(title, resized)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
'''
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to the source image")
ap.add_argument("-ssm", "--sourceskinmask", required=True, help="Path to the source skin mask image")
ap.add_argument("-t", "--target", required=True, help="Path to the target image")
ap.add_argument("-tsm", "--targetskinmask", required=True, help="Path to the source skin mask image")
ap.add_argument("-tbm", "--targetbodymask", required=True, help="Path to the source skin mask image")
ap.add_argument("-c", "--clip", type=str2bool, default='t', help="Should np.clip scale L*a*b* values before final conversion to BGR?")
ap.add_argument("-p", "--preservePaper", type=str2bool, default='t', help="Should color transfer strictly follow methodology laid out in original paper?")
ap.add_argument("-o", "--output", help="Path to the output image (optional)")
ap.add_argument("-tclth", "--targetcloth", required=True, help="Path to the clothing image of the target")
ap.add_argument("-d", "--debug", type=str2bool, default='f', help="Show debug information")
args = vars(ap.parse_args())

# Load the images
source = cv2.imread(args["source"])
target = cv2.imread(args["target"])
tcloth = cv2.imread(args["targetcloth"])
ssmask = cv2.imread(args["sourceskinmask"])
tsmask = cv2.imread(args["targetskinmask"])
tbmask = cv2.imread(args["targetbodymask"])
'''
# Load the images
imgpath = "D:\\work\\project\\VRC\\color transfer\\color-transfer0430\\data+4\\7"
source = cv2.imread(os.path.join(imgpath, "maleuser7_skin..jpg"))
target = cv2.imread(os.path.join(imgpath, "malemodel7_skin..jpg"))
tcloth = cv2.imread(os.path.join(imgpath, "malemodel7_noneskin..jpg"))
ssmask = cv2.imread(os.path.join(imgpath, "userskinmask.png"))
tsmask = cv2.imread(os.path.join(imgpath, "modelskinmask.png"))
tbmask = cv2.imread(os.path.join(imgpath, "modelbodymask.png"))

tsmask = tsmask*(tsmask > 128)
ssmask = ssmask*(ssmask > 128)
tbmask = tbmask*(tbmask > 128)
# Transfer the color distribution from the source image to the target image
'''
transfer = color_transfer(
    source=source,
    ssmask=ssmask,
    target=target,
    tsmask=tsmask,
    clip=args["clip"],
    preserve_paper=args["preservePaper"],
    debug=args["debug"]
)
'''
transfer = color_transfer(
    source=source,
    ssmask=ssmask[:,:,0],
    target=target,
    tsmask=tsmask[:,:,0],
    clip=True,
    preserve_paper=False,
    debug=False
)


# Create result image by combining:
# 1. Clothing from concat image (non-white regions)
# 2. Color-transferred skin from transfer image (white regions of concat)
result = tcloth.copy()
result[tsmask>0] = transfer[tsmask>0]  # Use transferred skin where concat is white
#cv2.imshow("result", result )
#cv2.waitKey(0)
# Save the result if an output path is provided
#if args["output"] is not None:
#    cv2.imwrite(args["output"], result)
cv2.imwrite(os.path.join(imgpath, "outputnew.jpg"), result)

# Show all images for comparison
# show_image("Source", source)
# show_image("Target", target)
# show_image("Transfer", transfer)
# show_image("Concat", concat)
show_image("Result", result)

cv2.waitKey(0)