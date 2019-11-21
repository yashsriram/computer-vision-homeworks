import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import math

NN_RATIO = 0.7
NUM_IAC_ITER = 125
ransac_thr = 10
ransac_iter = 1000
RANDOM_SEED = 42


# Matches SIFT features in template to nearest neighbours in target with a distance ratio threshold
def find_match(_template, _target):
    print('Finding and matching SIFT features with distance ratio filtering')
    # SIFT features (descriptors) extraction
    sift = cv2.xfeatures2d.SIFT_create()
    template_kps, template_descriptors = sift.detectAndCompute(_template, None)
    target_kps, target_descriptors = sift.detectAndCompute(_target, None)
    # Nearest neighbour matching
    model = NearestNeighbors(n_neighbors=2).fit(target_descriptors)
    distances, indices = model.kneighbors(template_descriptors)
    # Ratio culling
    x1 = []
    x2 = []
    # For each kp in img1 if nearest neighbour distance ratio <
    for i in range(len(template_kps)):
        d1, d2 = distances[i]
        if (d1 / d2) <= NN_RATIO:
            point1 = template_kps[i].pt
            point2 = target_kps[indices[i][0]].pt
            x1.append(point1)
            x2.append(point2)

    print('{} SIFT feature matches done from template to target with filtering ratio {}'.format(len(x1), NN_RATIO))
    return np.asarray(x1), np.asarray(x2)


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    print('Calculating initial affine transform using SIFT feature matches and RANSAC')
    print('RANSAC params: thres = {}, iter = {}'.format(ransac_thr, ransac_iter))
    best_affine_transform = None
    best_num_inliers = 0
    for i in range(ransac_iter):
        # Select 3 points in random
        random_index = np.random.choice(x1.shape[0], 3, replace=False)
        X1 = x1[random_index]
        X2 = x2[random_index]
        # Solve for affine transform
        A = np.array([
            [X1[0][0], X1[0][1], 1, 0, 0, 0],
            [0, 0, 0, X1[0][0], X1[0][1], 1],
            [X1[1][0], X1[1][1], 1, 0, 0, 0],
            [0, 0, 0, X1[1][0], X1[1][1], 1],
            [X1[2][0], X1[2][1], 1, 0, 0, 0],
            [0, 0, 0, X1[2][0], X1[2][1], 1],
        ])
        b = X2.reshape(-1)
        try:
            affine_transform = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        # Reshape affine transform matrix
        affine_transform = np.array(list(affine_transform) + [0, 0, 1])
        affine_transform = affine_transform.reshape((3, 3))
        if best_affine_transform is None:
            best_affine_transform = affine_transform
        # Calculate number of inliers
        num_inliers = 0
        for j in range(x1.shape[0]):
            template_point = np.array(list(x1[j]) + [1])
            target_point = np.array(list(x2[j]) + [1])
            template_point_image = np.matmul(affine_transform, template_point)
            distance = np.sqrt(np.sum((template_point_image - target_point) ** 2))
            if distance < ransac_thr:
                num_inliers += 1
        if num_inliers > best_num_inliers:
            best_affine_transform = affine_transform
            best_num_inliers = num_inliers

    print('For best affine transform model, #Inliers/Total  = {}/{}'.format(best_num_inliers, x1.shape[0]))

    return best_affine_transform


def warp_image(img, A, output_size):
    img_warped = np.zeros(output_size)
    r, c = output_size
    for ri in range(r):
        for ci in range(c):
            point_on_img = np.matmul(A, np.asarray([ci, ri, 1]))
            _X = np.array([math.ceil(point_on_img[0]) - point_on_img[0], point_on_img[0] - math.floor(point_on_img[0])])
            _M = np.array([
                [img[math.floor(point_on_img[1]), math.floor(point_on_img[0])],
                 img[math.ceil(point_on_img[1]), math.floor(point_on_img[0])]],
                [img[math.floor(point_on_img[1]), math.ceil(point_on_img[0])],
                 img[math.ceil(point_on_img[1]), math.ceil(point_on_img[0])]],
            ])
            _Y = np.array([math.ceil(point_on_img[1]) - point_on_img[1], point_on_img[1] - math.floor(point_on_img[1])])
            img_warped[ri, ci] = np.matmul(_X, np.matmul(_M, _Y.reshape(2, 1)))
    return img_warped


def get_differential_filter():
    filter_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
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


def get_dell(im_dx, im_dy):
    assert im_dx.shape == im_dy.shape
    m, n = im_dx.shape
    dell = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            dell[i][j] = [im_dx[i][j], im_dy[i][j]]
    return dell


def get_affine_transform(p):
    return np.array([
        [1 + p[0][0], p[1][0], p[2][0]],
        [p[3][0], 1 + p[4][0], p[5][0]],
        [0, 0, 1],
    ])


def align_image(template, target, A):
    print('Inverse compositional alignment')
    # Calculating dell I
    filter_x, filter_y = get_differential_filter()
    im_dx, im_dy = filter_image(template, filter_x), filter_image(template, filter_y)
    dell_template = get_dell(im_dx, im_dy)
    print('Calculated dell(template) with shape {}'.format(dell_template.shape))
    # Calculating steepest descent images
    steepest_descent_images = np.zeros((template.shape[0], template.shape[1], 6))
    for ri in range(template.shape[0]):
        for ci in range(template.shape[1]):
            steepest_descent_images[ri, ci] = np.matmul(
                dell_template[ri, ci],
                np.array([
                    [ci, ri, 1, 0, 0, 0],
                    [0, 0, 0, ci, ri, 1],
                ])
            )
    print('Calculated steepest descent images with shape {}'.format(steepest_descent_images.shape))
    # plt.subplot(231)
    # plt.imshow(steepest_descent_images[:, :, 0], cmap='hot')
    # plt.title('u*dx')
    # plt.axis('off')
    # plt.subplot(232)
    # plt.imshow(steepest_descent_images[:, :, 1], cmap='hot')
    # plt.title('v*dx')
    # plt.axis('off')
    # plt.subplot(233)
    # plt.imshow(steepest_descent_images[:, :, 2], cmap='hot')
    # plt.title('dx')
    # plt.axis('off')
    # plt.subplot(234)
    # plt.imshow(steepest_descent_images[:, :, 3], cmap='hot')
    # plt.title('u*dy')
    # plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(steepest_descent_images[:, :, 4], cmap='hot')
    # plt.title('v*dy')
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(steepest_descent_images[:, :, 5], cmap='hot')
    # plt.title('dy')
    # plt.axis('off')
    # plt.show()
    # Calulating Hessian
    hessian = np.zeros((6, 6))
    for ri in range(template.shape[0]):
        for ci in range(template.shape[1]):
            Hx = np.matmul(steepest_descent_images[ri, ci].reshape(6, 1), steepest_descent_images[ri, ci].reshape(1, 6))
            hessian += Hx
    hessian_inv = np.linalg.inv(hessian)
    print('Calculated hessian with shape {} and values\n{}'.format(hessian.shape, hessian))
    print('Calculated hessian inverse with shape {} and values\n{}'.format(hessian_inv.shape, hessian_inv))
    # Refining warp function (here affine transform)
    refined_A = A
    error_norms = []
    num_iterations = 0
    while True:
        target_warped = warp_image(target, refined_A, template.shape)
        Ierr = target_warped - template
        error_norm = np.sqrt(np.sum(Ierr ** 2))
        error_norms.append(error_norm)
        F = np.zeros((6, 1))
        for ri in range(template.shape[0]):
            for ci in range(template.shape[1]):
                F += (np.transpose(steepest_descent_images[ri, ci]) * Ierr[ri, ci]).reshape(6, 1)
        dell_p = np.matmul(hessian_inv, F)
        refined_A = np.matmul(refined_A, np.linalg.inv(get_affine_transform(dell_p)))
        print('Refining warp, iteration = {}, error_norm = {}'.format(num_iterations, error_norm))
        num_iterations += 1
        if num_iterations > NUM_IAC_ITER or error_norm < 1e3:
            break
    return refined_A, np.array(error_norms)


def track_multi_frames(original_template, target_list):
    # Initialize using feature vectors
    x1, x2 = find_match(original_template, target_list[0])
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    A_list = []
    template = original_template
    for i, target in enumerate(target_list):
        print('Aligning target number {}'.format(i))
        A, errors = align_image(template, target, A)
        template = warp_image(target, A, template.shape)
        A_list.append(A)

    return A_list


def visualize_affine_transform():
    # Plot original keypoints in target image
    plt.plot(x2[:, 0], x2[:, 1], 'ro')
    # Plot trasformed keypoints (from template) in target image
    for i in range(x2.shape[0]):
        x2_hat = np.matmul(A, np.array(list(x1[i]) + [1]))
        plt.plot(x2_hat[0], x2_hat[1], 'go')
    # Plotting boundaries of template image
    ul = np.matmul(A, np.array([0, 0, 1]))
    ur = np.matmul(A, np.array([template.shape[1], 0, 1]))
    ll = np.matmul(A, np.array([0, template.shape[0], 1]))
    lr = np.matmul(A, np.array([template.shape[1], template.shape[0], 1]))
    plt.plot([ul[0], ur[0]], [ul[1], ur[1]], 'b')
    plt.plot([ul[0], ur[0]], [ul[1], ur[1]], 'bo')
    plt.plot([lr[0], ur[0]], [lr[1], ur[1]], 'b')
    plt.plot([lr[0], ur[0]], [lr[1], ur[1]], 'bo')
    plt.plot([lr[0], ll[0]], [lr[1], ll[1]], 'b')
    plt.plot([lr[0], ll[0]], [lr[1], ll[1]], 'bo')
    plt.plot([ul[0], ll[0]], [ul[1], ll[1]], 'b')
    plt.plot([ul[0], ll[0]], [ul[1], ll[1]], 'bo')
    plt.imshow(target_list[0], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                          [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Initialize random seed
    print('Random seed = {}'.format(RANDOM_SEED))
    np.random.seed(RANDOM_SEED)

    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i + 1), 0)  # read as grey scale image
        target_list.append(target)

    # x1, x2 = find_match(template, target_list[0])
    # visualize_find_match(template, target_list[0], x1, x2)
    #
    # A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    # visualize_affine_transform()
    # img_warped = warp_image(target_list[0], A, template.shape)
    # img_diff = np.abs(template - img_warped)
    # error = np.sqrt(np.sum(img_diff ** 2))
    # print('Initial error b/w template and warped image = {}'.format(error))
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    # plt.imshow(img_diff, cmap='jet')
    # plt.show()
    #
    # A_refined, errors = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)
