import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    Calculate PSNR between two images.
   
    Args:
    img1 (torch.Tensor): The first image tensor (reference image).
    img2 (torch.Tensor): The second image tensor (reconstructed image).
    max_pixel_value (float): The maximum possible pixel value of the image. Typically 1.0 if images are normalized.
   
    Returns:
    float: The PSNR value.
    """
   
    # Ensure the input tensors are in float32
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
   
    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)
   
    if mse == 0:
        return float('inf')  # Perfect match
   
    # Compute PSNR
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
   
    return psnr.item()

def load_image_as_tensor(image_path):
    """
    Load an image from a file path and convert it to a PyTorch tensor.
   
    Args:
    image_path (str): Path to the image file.
   
    Returns:
    torch.Tensor: The image as a tensor.
    """
    # Open the image file
    img = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
   
    # Convert the image to a tensor and normalize pixel values to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts PIL Image to Tensor and scales pixel values [0, 255] to [0, 1]
    ])
   
    img_tensor = transform(img)
   
    return img_tensor

def calculate_psnr_for_image_pairs_by_index(dir1, dir2):
    """
    Calculate PSNR between corresponding images in two directories based on their index, not file name.
   
    Args:
    dir1 (str): Directory path for the first set of images (reference images).
    dir2 (str): Directory path for the second set of images (reconstructed images).
   
    Returns:
    None
    """
    # List all files in both directories
    images1 = sorted(os.listdir(dir1))
    images2 = sorted(os.listdir(dir2))
   
    # Ensure both directories have the same number of images
    if len(images1) != len(images2):
        raise ValueError("The two directories must contain the same number of images!")
   
    psnr_values = []
    psnr_values2 = []
   
    for i in range(len(images1)):
        image_path1 = os.path.join(dir1, images1[i])
        image_path2 = os.path.join(dir2, images2[i])
       
        # Load images as tensors
        img1 = load_image_as_tensor(image_path1)
        img2 = load_image_as_tensor(image_path2)
       
        # Ensure both images have the same size (necessary for PSNR calculation)
        if img1.size() != img2.size():
            raise ValueError(f"The images at index {i} (files {images1[i]} and {images2[i]}) must have the same dimensions!")
       
        # Calculate PSNR for the current pair of images
        psnr_value = calculate_psnr(img1, img2)
        psnr_values.append(psnr_value)
        print(f"PSNR for image pair {i}: {psnr_value} dB")
   
    return psnr_values


def calculate_ssim_for_image_pairs_by_index(dir1, dir2):
    """
    Calculate SSIM between corresponding images in two directories based on their index, not file name.
    
    Args:
    dir1 (str): Directory path for the first set of images (reference images).
    dir2 (str): Directory path for the second set of images (reconstructed images).
    
    Returns:
    List of SSIM values for each image pair.
    """
    # List all files in both directories
    images1 = sorted(os.listdir(dir1))
    images2 = sorted(os.listdir(dir2))
    
    # Ensure both directories have the same number of images
    if len(images1) != len(images2):
        raise ValueError("The two directories must contain the same number of images!")
    
    ssim_values = []
    
    for i in range(len(images1)):
        image_path1 = os.path.join(dir1, images1[i])
        image_path2 = os.path.join(dir2, images2[i])
        
        # Load images as tensors
        img1 = load_image_as_tensor(image_path1)
        img2 = load_image_as_tensor(image_path2)
        
        # Ensure both images have the same size (necessary for SSIM calculation)
        if img1.size() != img2.size():
            raise ValueError(f"The images at index {i} (files {images1[i]} and {images2[i]}) must have the same dimensions!")
        
        # Convert tensors to NumPy arrays for SSIM calculation
        img1_np = img1.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
        img2_np = img2.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)

        win_size = min(7, min_dimension)
        
        # Calculate SSIM for the current pair of images
        ssim_value, _ = ssim(img1_np, img2_np, multichannel=True, full=True)
        ssim_values.append(ssim_value)
        print(f"SSIM for image pair {i}: {ssim_value}")
    
    return ssim_values

# Example usage:
dir1 = '../renders/DroneLakeLag/CombinedVideo/Nerfacto100Percent'  # Replace with the directory containing the first set of images
dir2 = '../renders/DroneLakeLag/CombinedVideo/Nerfacto90Percent'  # Replace with the directory containing the second set of images
dir3 = '../renders/DroneLakeLag/CombinedVideo/Nerfacto10Percent'

# Calculate PSNR for all image pairs based on their index in the directory
#psnr_values = calculate_psnr_for_image_pairs_by_index(dir1, dir2)
#psnr_values2 = calculate_psnr_for_image_pairs_by_index(dir1, dir3)

ssim_values = calculate_ssim_for_image_pairs_by_index(dir1, dir2)
ssim_values2 = calculate_ssim_for_image_pairs_by_index(dir1, dir3)

# calculate and print the medians
median_1 = np.median(ssim_values)
median_2 = np.median(ssim_values2)

print(f"Median of Dataset 1: {median_1}")
print(f"Median of Dataset 2: {median_2}")

# Combine the two sets of PSNR values into a list
data = [ssim_values, ssim_values2]

# Create the box and whisker plots
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['10 percent downsampled', '90 percent downsampled'])

# Add titles and labels
plt.title("Box and Whisker Plot of SSIM Values between non-downsampled render and downsampled render")
plt.ylabel("SSIMis (dB)")

# Display the plot
plt.show()

# Print final SSIM results
print("SSIM values for all image pairs:", ssim_values)

# Print final PSNR results
#print("PSNR values for all image pairs:", psnr_values)