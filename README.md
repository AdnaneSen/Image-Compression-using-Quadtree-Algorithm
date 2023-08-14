# Image-Compression-using-Quadtree-Algorithm

This repository contains Python code for compressing images using the quadtree algorithm. The quadtree algorithm recursively divides an image into regions of similar color, reducing the number of pixels required to represent the image while preserving visual quality. The code provided implements both explicit and implicit quadtree-based image compression methods.

# Features

    - Explicit Quadtree Compression: The code includes an explicit quadtree-based compression method that partitions the image into rectangular regions and constructs a tree structure representing these regions. Homogeneous regions are represented by single pixels, and non-homogeneous regions are subdivided further.

    - Implicit Quadtree Compression: The code also implements an implicit quadtree-based compression method. Instead of explicitly constructing a tree, this method uses a list representation where each element corresponds to a node in the tree. This approach is memory-efficient and can be more performant for large images.

    - PSNR Calculation: The code provides functions to calculate the Peak Signal-to-Noise Ratio (PSNR) for both explicit and implicit quadtree-based compressed images. PSNR is a measure of image quality that compares the original and compressed images, quantifying the amount of information lost during compression.

    - Image Blurring: An additional image blurring function is included, utilizing a Gaussian blur filter to improve image compression quality by reducing high-frequency noise.

# Usage
   The code will:
        Generate an explicit quadtree structure and compress the image.
        Calculate the PSNR for the explicit compression.
        Generate an implicit quadtree structure and compress the image.
        Calculate the PSNR for the implicit compression.
        Display the compressed image and save it.
