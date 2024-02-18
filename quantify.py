
########################################################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def compute_image_statistics(image):
    # Compute mean, standard deviation, and variance of pixel values
    mean_value = np.mean(image)
    std_dev = np.std(image)
    variance = np.var(image)
    return mean_value, std_dev, variance


def compute_image_difference(image1, image2):
    # Compute statistics for both images
    mean1, std_dev1, var1 = compute_image_statistics(image1)
    mean2, std_dev2, var2 = compute_image_statistics(image2)

    # Compute differences between statistics
    mean_diff = abs(mean1 - mean2)
    std_dev_diff = abs(std_dev1 - std_dev2)
    var_diff = abs(var1 - var2)

    return mean_diff, std_dev_diff, var_diff


def visualize_difference(image1, image2, threshold):
    # Compute differences between images
    mean_diff, std_dev_diff, var_diff = compute_image_difference(image1, image2)

    # Apply threshold to determine significant differences
    significant_diff_mask = np.logical_or(mean_diff > threshold, std_dev_diff > threshold, var_diff > threshold)

    # Create a heatmap of significant differences
    heatmap = np.zeros_like(image1, dtype=np.uint8)
    heatmap[significant_diff_mask] = 255  # Highlight significant differences

    # Scale the heatmap to enhance visualization
    scaled_heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return scaled_heatmap


def compute_histogram_intersection(hist1, hist2):
    # Compute histogram intersection distance
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)


def compute_mean_squared_error(image1, image2):
    # Compute Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)
    return mse


def compute_ssim(image1, image2):
    # Compute Structural Similarity Index (SSIM)
    return ssim(image1, image2)


def compute_sharpness(image):
    # Compute gradient magnitude using Sobel filter
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    sharpness = np.mean(gradient_magnitude)
    return sharpness


def compute_structural_content(image1, image2):
    # Compute SSIM focusing on structural content
    return ssim(image1, image2, data_range=image1.max() - image1.min(), gaussian_weights=True,
                use_sample_covariance=False)


def compute_psnr(image1, image2):
    # Compute Peak Signal-to-Noise Ratio (PSNR)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(image1)
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_natural_scene_statistics(image):
    # Compute natural scene statistics
    pixel_intensity_distribution = np.histogram(image.flatten(), bins=256, range=[0, 256])[0] / (
                image.shape[0] * image.shape[1])
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_distribution = np.histogram(gradient_magnitude.flatten(), bins=256, range=[0, 256])[0] / (
                image.shape[0] * image.shape[1])
    return pixel_intensity_distribution, gradient_distribution


def main():
    # Load two images
    image1 = cv2.imread('bag_68.jpeg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('bag_130.jpeg', cv2.IMREAD_GRAYSCALE)

    # Compute histograms for both images
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)

    # Compute histogram intersection distance
    hist_diff = compute_histogram_intersection(hist1, hist2)

    # Compute Mean Squared Error (MSE)
    mse = compute_mean_squared_error(image1, image2)

    # Compute Structural Similarity Index (SSIM)
    ssim_score = compute_ssim(image1, image2)

    # Set threshold for significant differences
    threshold = 0.1  # Adjust as needed

    # Compute sharpness for both images
    sharpness_image1 = compute_sharpness(image1)
    sharpness_image2 = compute_sharpness(image2)

    # Compute structural content
    structural_content = compute_structural_content(image1, image2)

    # Compute fidelity (PSNR)
    psnr = compute_psnr(image1, image2)

    # Compute natural scene statistics
    intensity_dist1, gradient_dist1 = compute_natural_scene_statistics(image1)
    intensity_dist2, gradient_dist2 = compute_natural_scene_statistics(image2)

    # Compute image statistics
    mean1, std_dev1, var1 = compute_image_statistics(image1)
    mean2, std_dev2, var2 = compute_image_statistics(image2)

    # Visualize differences between images
    # heatmap = visualize_difference(image1, image2, threshold)

    # Display images, heatmap, histograms, and metrics
    plt.figure(figsize=(28, 12))

    plt.subplot(2, 7, 1)
    plt.title('Image 1')
    plt.imshow(image1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 7, 2)
    plt.title('Image 2')
    plt.imshow(image2, cmap='gray')
    plt.axis('off')

    # plt.subplot(2, 7, 3)
    # plt.title('Difference Heatmap')
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.axis('off')

    plt.subplot(2, 7, 4)
    plt.title('Histogram Difference: {:.2f}\nMSE: {:.2f}'.format(hist_diff, mse))
    plt.plot(hist1, color='blue', label='Image 1')
    plt.plot(hist2, color='red', label='Image 2')
    plt.legend()
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 7, 5)
    plt.title('SSIM: {:.4f}'.format(ssim_score))
    plt.imshow(cv2.cvtColor(cv2.absdiff(image1, image2), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 7, 6)
    plt.title('Sharpness (Image 1): {:.2f}'.format(sharpness_image1))
    plt.axis('off')

    plt.subplot(2, 7, 7)
    plt.title('Sharpness (Image 2): {:.2f}'.format(sharpness_image2))
    plt.axis('off')

    plt.subplot(2, 7, 8)
    plt.title('Structural Content: {:.4f}'.format(structural_content))
    plt.axis('off')

    plt.subplot(2, 7, 9)
    plt.title('Fidelity (PSNR): {:.2f}'.format(psnr))
    plt.axis('off')

    plt.subplot(2, 7, 10)
    plt.title('Intensity Distribution (Image 1)')
    plt.plot(intensity_dist1, color='blue', label='Image 1')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 7, 11)
    plt.title('Intensity Distribution (Image 2)')
    plt.plot(intensity_dist2, color='red', label='Image 2')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 7, 12)
    plt.title('Gradient Distribution (Image 1)')
    plt.plot(gradient_dist1, color='blue', label='Image 1')
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 7, 13)
    plt.title('Gradient Distribution (Image 2)')
    plt.plot(gradient_dist2, color='red', label='Image 2')
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')
    plt.legend()

    # Add a subplot for mean, standard deviation, and variance
    plt.subplot(2, 7, (14, 16))
    plt.title('Image Statistics')
    plt.text(0.1, 0.8,
             f"Mean (Image 1): {mean1:.2f}\nStd Dev (Image 1): {std_dev1:.2f}\nVariance (Image 1): {var1:.2f}",
             fontsize=10, verticalalignment='top')
    plt.text(0.1, 0.6,
             f"Mean (Image 2): {mean2:.2f}\nStd Dev (Image 2): {std_dev2:.2f}\nVariance (Image 2): {var2:.2f}",
             fontsize=10, verticalalignment='top')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

