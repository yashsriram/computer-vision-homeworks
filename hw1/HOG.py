import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
	filter_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	filter_y = np.transpose(filter_x)
	return filter_x, filter_y


def filter_image(im, filter):
	padded = np.pad(im, 1)
	m, n = im.shape
	u, v = filter.shape
	im_filtered = np.zeros((m, n))
	for i in range(m):
		for j in range(n):
			pixel = 0
			for x in range(u):
				for y in range(v):
					pixel += padded[i + x][j + y] * filter[x][y]
			im_filtered[i][j] = pixel
	return im_filtered


def get_gradient(im_dx, im_dy):
	assert im_dx.shape == im_dy.shape
	m, n = im_dx.shape
	grad_mag = np.zeros((m, n))
	grad_angle = np.zeros((m, n))
	for i in range(m):
		for j in range(n):
			grad_mag[i][j] = np.sqrt(im_dx[i][j] ** 2 + im_dy[i][j] ** 2)
			grad_angle[i][j] = np.arctan2(im_dy[i][j], im_dx[i][j])
	return grad_mag, grad_angle


def get_angle_bin(angle):
	angle_deg = angle * 180 / np.pi
	if 0 <= angle_deg < 15 or 165 <= angle_deg < 180:
		return 0
	elif 15 <= angle_deg < 45:
		return 1
	elif 45 <= angle_deg < 75:
		return 2
	elif 75 <= angle_deg < 105:
		return 3
	elif 105 <= angle_deg < 135:
		return 4
	elif 135 <= angle_deg < 165:
		return 5

def build_histogram(grad_mag, grad_angle, cell_size):
	m, n = grad_mag.shape
	M = int(m / cell_size)
	N = int(n / cell_size)
	ori_histo = np.zeros((M, N, 6))
	for i in range(M):
		for j in range(N):
			for x in range(cell_size):
				for y in range(cell_size):
					angle = grad_angle[i * cell_size + x][j * cell_size + y]
					mag = grad_mag[i * cell_size + x][j * cell_size + y]
					bin = get_angle_bin(angle)
					ori_histo[i][j][bin] += mag

	return ori_histo


def get_block_descriptor(ori_histo, block_size):
	M, N, bins = ori_histo.shape
	e = 0.001
	ori_histo_normalized = np.zeros((M - block_size + 1, N - block_size + 1, bins * block_size * block_size))
	for i in range(M - block_size + 1):
		for j in range(N - block_size + 1):
			unnormalized = []
			for x in range(block_size):
				for y in range(block_size):
					for z in range(bins):
						unnormalized.append(ori_histo[i + x][j + y][z])
			unnormalized = np.asarray(unnormalized)
			den = np.sqrt(np.sum(unnormalized ** 2) + e ** 2)
			normalized = unnormalized / den
			for p in range(bins * block_size * block_size):
				ori_histo_normalized[i][j][p] = normalized[p]
	
	return ori_histo_normalized


def extract_hog(im):
	# convert grey-scale image to double format
	im = im.astype('float') / 255.0
	filter_x, filter_y = get_differential_filter()
	im_dx, im_dy = filter_image(im, filter_x), filter_image(im, filter_y)
	grad_mag, grad_angle = get_gradient(im_dx, im_dy)
	ori_histo = build_histogram(grad_mag, grad_angle, 8)
	ori_histo_normalized = get_block_descriptor(ori_histo, 2)
	hog = ori_histo_normalized.reshape((-1))

	plt.imshow(im, cmap='gray')
	plt.show()
	plt.imshow(im_dx, cmap='hot', interpolation='nearest')
	plt.show()
	plt.imshow(im_dy, cmap='hot', interpolation='nearest')
	plt.show()
	plt.imshow(grad_mag, cmap='hot', interpolation='nearest')
	plt.show()
	plt.imshow(grad_angle, cmap='hot', interpolation='nearest')
	plt.show()

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


