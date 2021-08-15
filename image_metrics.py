#!/usr/bin/env python3

import torch
import piq
import cv2
import sys


@torch.no_grad()
def main(img1, img2):
    # Read RGB image and it's noisy version
    x = torch.tensor(cv2.imread(img1)).permute(2, 0, 1)[None, ...] / 255.
    y = torch.tensor(cv2.imread(img2)).permute(2, 0, 1)[None, ...] / 255.
    
    # To compute BRISQUE score as a measure, use lower case function from the library
    brisque_index: torch.Tensor = piq.brisque(x, data_range=1., reduction='none')
    # In order to use BRISQUE as a loss function, use corresponding PyTorch module.
    # Note: the back propagation is not available using torch==1.5.0.
    # Update the environment with latest torch and torchvision.
    brisque_loss: torch.Tensor = piq.BRISQUELoss(data_range=1., reduction='none')(x)

    # To compute Content score as a loss function, use corresponding PyTorch module
    # By default VGG16 model is used, but any feature extractor model is supported.
    # Don't forget to adjust layers names accordingly. Features from different layers can be weighted differently.
    # Use weights parameter. See other options in class docstring.
    content_loss = piq.ContentLoss(
        feature_extractor="vgg16", layers=("relu3_3", ), reduction='none')(x, y)

    # To compute DISTS as a loss function, use corresponding PyTorch module
    # By default input images are normalized with ImageNet statistics before forwarding through VGG16 model.
    # If there is no need to normalize the data, use mean=[0.0, 0.0, 0.0] and std=[1.0, 1.0, 1.0].
    dists_loss = piq.DISTS(reduction='none')(x, y)

    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(x, y, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    fsim_loss = piq.FSIMLoss(data_range=1., reduction='none')(x, y)

    # To compute GMSD as a measure, use lower case function from the library
    # This is port of MATLAB version from the authors of original paper.
    # In any case it should me minimized. Usually values of GMSD lie in [0, 0.35] interval.
    gmsd_index: torch.Tensor = piq.gmsd(x, y, data_range=1., reduction='none')
    # In order to use GMSD as a loss function, use corresponding PyTorch module:
    gmsd_loss: torch.Tensor = piq.GMSDLoss(data_range=1., reduction='none')(x, y)

    # To compute HaarPSI as a measure, use lower case function from the library
    # This is port of MATLAB version from the authors of original paper.
    haarpsi_index: torch.Tensor = piq.haarpsi(x, y, data_range=1., reduction='none')
    # In order to use HaarPSI as a loss function, use corresponding PyTorch module
    haarpsi_loss: torch.Tensor = piq.HaarPSILoss(data_range=1., reduction='none')(x, y)

    # To compute LPIPS as a loss function, use corresponding PyTorch module
    lpips_loss: torch.Tensor = piq.LPIPS(reduction='none')(x, y)

    # To compute MDSI as a measure, use lower case function from the library
    mdsi_index: torch.Tensor = piq.mdsi(x, y, data_range=1., reduction='none')
    # In order to use MDSI as a loss function, use corresponding PyTorch module
    mdsi_loss: torch.Tensor = piq.MDSILoss(data_range=1., reduction='none')(x, y)

    # To compute MS-SSIM index as a measure, use lower case function from the library:
    ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(x, y, data_range=1.)
    # In order to use MS-SSIM as a loss function, use corresponding PyTorch module:
    ms_ssim_loss = piq.MultiScaleSSIMLoss(data_range=1., reduction='none')(x, y)

    # To compute Multi-Scale GMSD as a measure, use lower case function from the library
    # It can be used both as a measure and as a loss function. In any case it should me minimized.
    # By defualt scale weights are initialized with values from the paper.
    # You can change them by passing a list of 4 variables to scale_weights argument during initialization
    # Note that input tensors should contain images with height and width equal 2 ** number_of_scales + 1 at least.
    ms_gmsd_index: torch.Tensor = piq.multi_scale_gmsd(
        x, y, data_range=1., chromatic=True, reduction='none')
    # In order to use Multi-Scale GMSD as a loss function, use corresponding PyTorch module
    ms_gmsd_loss: torch.Tensor = piq.MultiScaleGMSDLoss(
        chromatic=True, data_range=1., reduction='none')(x, y)

    # To compute PSNR as a measure, use lower case function from the library.
    psnr_index = piq.psnr(x, y, data_range=1., reduction='none')
    
    # To compute PieAPP as a loss function, use corresponding PyTorch module:
    pieapp_loss: torch.Tensor = piq.PieAPP(reduction='none', stride=32)(x, y)

    # To compute SSIM index as a measure, use lower case function from the library:
    ssim_index = piq.ssim(x, y, data_range=1.)
    # In order to use SSIM as a loss function, use corresponding PyTorch module:
    ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(x, y)

    # To compute Style score as a loss function, use corresponding PyTorch module:
    # By default VGG16 model is used, but any feature extractor model is supported.
    # Don't forget to adjust layers names accordingly. Features from different layers can be weighted differently.
    # Use weights parameter. See other options in class docstring.
    style_loss = piq.StyleLoss(feature_extractor="vgg16", layers=("relu3_3", ))(x, y)

    # To compute TV as a measure, use lower case function from the library:
    tv_index: torch.Tensor = piq.total_variation(x)
    # In order to use TV as a loss function, use corresponding PyTorch module:
    tv_loss: torch.Tensor = piq.TVLoss(reduction='none')(x)

    # To compute VIF as a measure, use lower case function from the library:
    vif_index: torch.Tensor = piq.vif_p(x, y, data_range=1.)
    # In order to use VIF as a loss function, use corresponding PyTorch class:
    vif_loss: torch.Tensor = piq.VIFLoss(sigma_n_sq=2.0, data_range=1.)(x, y)

    # To compute VSI score as a measure, use lower case function from the library:
    vsi_index: torch.Tensor = piq.vsi(x, y, data_range=1.)
    # In order to use VSI as a loss function, use corresponding PyTorch module:
    vsi_loss: torch.Tensor = piq.VSILoss(data_range=1.)(x, y)

    return {
            "brisque_index": brisque_index.item(), 
            "brisque_loss": brisque_loss.item(), 
            "content_loss": content_loss.item(), 
            "dists_loss": dists_loss.item(), 
            "fsim_index": fsim_index.item(), 
            "fsim_loss": fsim_loss.item(), 
            "gmsd_index": gmsd_index.item(), 
            "gmsd_loss": gmsd_loss.item(), 
            "haarpsi_index": haarpsi_index.item(), 
            "haarpsi_loss": haarpsi_loss.item(), 
            "lpips_loss": lpips_loss.item(), 
            "mdsi_index": mdsi_index.item(), 
            "mdsi_loss": mdsi_loss.item(), 
            "ms_ssim_index": ms_ssim_index.item(), 
            "ms_ssim_loss": ms_ssim_loss.item(),
            "ms_gmsd_index": ms_gmsd_index.item(),
            "ms_gmsd_loss": ms_gmsd_loss.item(),
            "psnr_index": psnr_index.item(),
            "pieapp_loss": pieapp_loss.item(),
            "ssim_index": ssim_index.item(),
            "ssim_loss": ssim_loss.item(),
            "style_loss": style_loss.item(),
            "tv_index": tv_index.item(),
            "tv_loss": tv_loss.item(),
            "vif_index": vif_index.item(),
            "vif_loss": vif_loss.item(),
            "vsi_index": vsi_index.item(),
            "vsi_loss": vsi_loss.item()
            }

def printMetrics(metrics):
    print("BRISQUE index: %0.4f, loss: %0.4f" %(metrics["brisque_index"], metrics["brisque_loss"]))
    print("ContentLoss: %0.4f" % (metrics["content_loss"]))
    print("DISTS: %0.4f" % (metrics["dists_loss"]))
    print("FSIM index: %0.4f, loss: %0.4f" % (metrics["fsim_index"], metrics["fsim_loss"]))
    print("GMSD index: %0.4f, loss: %0.4f" % (metrics["gmsd_index"], metrics["gmsd_loss"]))
    print("HaarPSI index: %0.4f, loss: %0.4f" % (metrics["haarpsi_index"], metrics["haarpsi_loss"]))
    print("LPIPS: %0.4f" % (metrics["lpips_loss"]))
    print("MDSI index: %0.4f, loss: %0.4f" % (metrics["mdsi_index"], metrics["mdsi_loss"]))
    print("MS-SSIM index: %0.4f, loss: %0.4f" % (metrics["ms_ssim_index"], metrics["ms_ssim_loss"]))
    print("MS-GMSDc index: %0.4f, loss: %0.4f" % (metrics["ms_gmsd_index"], metrics["ms_gmsd_loss"]))
    print("PSNR index: %0.4f" % (metrics["psnr_index"]))
    print("PieAPP loss: %0.4f" % (metrics["pieapp_loss"]))
    print("SSIM index: %0.4f, loss: %0.4f" % (metrics["ssim_index"], metrics["ssim_loss"]))
    print("Style: %0.4f" % (metrics["style_loss"]))
    print("TV index: %0.4f, loss: %0.4f" % (metrics["tv_index"], metrics["tv_loss"]))
    print("VIFp index: %0.4f, loss: %0.4f" % (metrics["vif_index"], metrics["vif_loss"]))
    print("VSI index: %0.4f, loss: %0.4f" % (metrics["vsi_index"], metrics["vsi_loss"]))

if __name__ == '__main__':
    #orig = '/media/home/alice/Photos/preview/2003/01/17/20030117_1_320_0.ppm'
    #jpg = '/media/home/alice/Photos/preview/2003/01/17/20030117_1_320_0.jpg'
    #webp = '/media/home/alice/Photos/preview/2003/01/17/20030117_1_320_0.webp'
    orig = sys.argv[1]
    jpg = sys.argv[2]
    webp = sys.argv[3]
    print("----- compare %s - %s" % (orig, jpg))
    metrics = main(orig, jpg)
    printMetrics(metrics)
    print("----- compare %s - %s" % (orig, webp))
    metrics = main(orig, webp)
    printMetrics(metrics)
