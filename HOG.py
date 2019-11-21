import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_differential_filter():
    filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return filter_x, filter_y


def filter_image(im, filter):
    m, n = im.shape
    filter_size, _ = filter.shape
    padded = np.pad(im, int(filter_size / 2))
    im_filtered = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            pixel = 0
            for x in range(filter_size):
                for y in range(filter_size):
                    pixel += filter[x][y] * padded[i + x][j + y]
            im_filtered[i][j] = pixel
    return im_filtered


def get_gradient(im_dx, im_dy):
    assert (im_dx.shape == im_dy.shape)
    m, n = im_dx.shape
    grad_mag = np.zeros((m, n))
    grad_angle = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            grad_mag[i][j] = np.sqrt(im_dx[i][j] ** 2 + im_dy[i][j] ** 2)
            grad_angle[i][j] = np.arctan2(im_dy[i][j], im_dx[i][j])
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    m, n = grad_mag.shape
    M, N = int(m / cell_size), int(n / cell_size)
    ori_histo = np.zeros((M, N, 6))
    for i in range(M):
        for j in range(N):
            for x in range(cell_size):
                for y in range(cell_size):
                    angle = grad_angle[M * cell_size + x][N * cell_size + y]
                    angle = angle * 180 / math.pi
                    bin = int(angle / 30)
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # print(im_dx.shape, im_dy.shape)
    # plt.imshow(np.abs(im_dx), cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(np.abs(im_dy), cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(np.abs(grad_mag), cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(np.abs(grad_angle), cmap='hot', interpolation='nearest')
    # plt.show()
    exit(0)

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)


