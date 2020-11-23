import random
import cv2
import os
from os import path as osp
from tqdm import tqdm
import numpy as np
import math


class GeometricPrimitives(object):
    def __init__(self, seed=1984):
        self.random_state = np.random.RandomState(seed)

    def get_random_color(self, background_color):
        """ Output a random scalar in grayscale with a least a small
            contrast with the background color """
        color = self.random_state.randint(256)
        if abs(color - background_color) < 30:  # not enough contrast
            color = (color + 128) % 256
        return color

    def get_different_color(self, previous_colors, min_dist=50, max_count=20):
        """ Output a color that contrasts with the previous colors
        Parameters:
          previous_colors: np.array of the previous colors
          min_dist: the difference between the new color and
                    the previous colors must be at least min_dist
          max_count: maximal number of iterations
        """
        color = self.random_state.randint(256)
        count = 0
        while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
            count += 1
            color = self.random_state.randint(256)
        return color

    def generate_background(self, size=(960, 1280), nb_blobs=100, min_rad_ratio=0.02,
                            max_rad_ratio=0.031, min_kernel_size=150, max_kernel_size=500):
        """ Generate a customized background image
        Parameters:
          size: size of the image
          nb_blobs: number of circles to draw
          min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
          max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
          min_kernel_size: minimal size of the kernel
          max_kernel_size: maximal size of the kernel
        """
        img = np.zeros(size, dtype=np.uint8)
        dim = max(size)
        cv2.randu(img, 0, 255)
        cv2.threshold(img, self.random_state.randint(256), 255, cv2.THRESH_BINARY, img)
        background_color = int(np.mean(img))
        blobs = np.concatenate([self.random_state.randint(0, size[1], size=(nb_blobs, 1)),
                                self.random_state.randint(0, size[0], size=(nb_blobs, 1))],
                               axis=1)
        for i in range(nb_blobs):
            col = self.get_random_color(background_color)
            cv2.circle(img, (blobs[i][0], blobs[i][1]),
                      np.random.randint(int(dim * min_rad_ratio),
                                        int(dim * max_rad_ratio)),
                      col, -1)
        kernel_size = self.random_state.randint(min_kernel_size, max_kernel_size)
        cv2.blur(img, (kernel_size, kernel_size), img)
        return img

    @staticmethod
    def _ccw(A, B, C, dim):
        """ Check if the points are listed in counter-clockwise order """
        if dim == 2:  # only 2 dimensions
            return ((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
                    > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
        else:  # dim should be equal to 3
            return ((C[:, 1, :] - A[:, 1, :])
                    * (B[:, 0, :] - A[:, 0, :])
                    > (B[:, 1, :] - A[:, 1, :])
                    * (C[:, 0, :] - A[:, 0, :]))

    @staticmethod
    def _intersect(A, B, C, D, dim):
        """ Return true if line segments AB and CD intersect """
        return np.any((GeometricPrimitives._ccw(A, C, D, dim) != GeometricPrimitives._ccw(B, C, D, dim)) &
                      (GeometricPrimitives._ccw(A, B, C, dim) != GeometricPrimitives._ccw(A, B, D, dim)))

    @staticmethod
    def _angle_between_vectors(v1, v2):
        """ Compute the angle (in rad) between the two vectors v1 and v2. """
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def draw_lines(self, img, nb_lines=10):
        """ Draw random lines and output the positions of the endpoints
        Parameters:
          nb_lines: maximal number of lines
        """
        num_lines = self.random_state.randint(1, nb_lines)
        segments = np.empty((0, 4), dtype=np.int)
        points = np.empty((0, 2), dtype=np.int)
        background_color = int(np.mean(img))
        min_dim = min(img.shape)
        for i in range(num_lines):
            x1 = self.random_state.randint(img.shape[1])
            y1 = self.random_state.randint(img.shape[0])
            p1 = np.array([[x1, y1]])
            x2 = self.random_state.randint(img.shape[1])
            y2 = self.random_state.randint(img.shape[0])
            p2 = np.array([[x2, y2]])
            # Check that there is no overlap
            if GeometricPrimitives._intersect(segments[:, 0:2], segments[:, 2:4], p1, p2, 2):
                continue
            segments = np.concatenate([segments, np.array([[x1, y1, x2, y2]])], axis=0)
            col = self.get_random_color(background_color)
            thickness = self.random_state.randint(min_dim * 0.01, min_dim * 0.02)
            cv2.line(img, (x1, y1), (x2, y2), col, thickness)
            points = np.concatenate([points, np.array([[x1, y1], [x2, y2]])], axis=0)
        return img, points

    def draw_polygon(self, img, max_sides=8):
        """ Draw a polygon with a random number of corners
        and return the corner points
        Parameters:
          max_sides: maximal number of sides + 1
        """
        num_corners = self.random_state.randint(3, max_sides)
        min_dim = min(img.shape[0], img.shape[1])
        rad = max(self.random_state.rand() * min_dim / 2, min_dim / 10)
        x = self.random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
        y = self.random_state.randint(rad, img.shape[0] - rad)

        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * math.pi, num_corners + 1)
        angles = [slices[i] + self.random_state.rand() * (slices[i + 1] - slices[i])
                  for i in range(num_corners)]
        points = np.array([[int(x + max(self.random_state.rand(), 0.4) * rad * math.cos(a)),
                            int(y + max(self.random_state.rand(), 0.4) * rad * math.sin(a))]
                           for a in angles])

        # Filter the points that are too close or that have an angle too flat
        norms = [np.linalg.norm(points[(i - 1) % num_corners, :]
                                - points[i, :]) for i in range(num_corners)]
        mask = np.array(norms) > 0.01
        points = points[mask, :]
        num_corners = points.shape[0]
        corner_angles = [GeometricPrimitives._angle_between_vectors(points[(i - 1) % num_corners, :] -
                                                                    points[i, :],
                                                                    points[(i + 1) % num_corners, :] -
                                                                    points[i, :])
                         for i in range(num_corners)]
        mask = np.array(corner_angles) < (2 * math.pi / 3)
        points = points[mask, :]
        num_corners = points.shape[0]
        if num_corners < 3:  # not enough corners
            return self.draw_polygon(img, max_sides)

        corners = points.reshape((-1, 1, 2))
        col = self.get_random_color(int(np.mean(img)))
        cv2.fillPoly(img, [corners], col)
        return img, points

    def draw_ellipses(self, img, nb_ellipses=20):
        """ Draw several ellipses
        Parameters:
          nb_ellipses: maximal number of ellipses
        """
        centers = np.empty((0, 2), dtype=np.int)
        rads = np.empty((0, 1), dtype=np.int)
        min_dim = min(img.shape[0], img.shape[1]) / 4
        background_color = int(np.mean(img))
        for i in range(nb_ellipses):
            ax = int(max(self.random_state.rand() * min_dim, min_dim / 5))
            ay = int(max(self.random_state.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = self.random_state.randint(max_rad, img.shape[1] - max_rad)  # center
            y = self.random_state.randint(max_rad, img.shape[0] - max_rad)
            new_center = np.array([[x, y]])

            # Check that the ellipsis will not overlap with pre-existing shapes
            diff = centers - new_center
            if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
                continue
            centers = np.concatenate([centers, new_center], axis=0)
            rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

            col = self.get_random_color(background_color)
            angle = self.random_state.rand() * 90
            cv2.ellipse(img, (x, y), (ax, ay), angle, 0, 360, col, -1)
        return img, np.empty((0, 2), dtype=np.int)

    def draw_star(self, img, nb_branches=6):
        """ Draw a star and output the interest points
        Parameters:
          nb_branches: number of branches of the star
        """
        num_branches = self.random_state.randint(3, nb_branches)
        min_dim = min(img.shape[0], img.shape[1])
        thickness = self.random_state.randint(min_dim * 0.01, min_dim * 0.02)
        rad = max(self.random_state.rand() * min_dim / 2, min_dim / 5)
        x = self.random_state.randint(rad, img.shape[1] - rad)  # select the center of a circle
        y = self.random_state.randint(rad, img.shape[0] - rad)
        # Sample num_branches points inside the circle
        slices = np.linspace(0, 2 * math.pi, num_branches + 1)
        angles = [slices[i] + self.random_state.rand() * (slices[i + 1] - slices[i])
                  for i in range(num_branches)]
        points = np.array([[int(x + max(self.random_state.rand(), 0.3) * rad * math.cos(a)),
                            int(y + max(self.random_state.rand(), 0.3) * rad * math.sin(a))]
                           for a in angles])
        points = np.concatenate(([[x, y]], points), axis=0)
        background_color = int(np.mean(img))
        for i in range(1, num_branches + 1):
            col = self.get_random_color(background_color)
            cv2.line(img,
                     (points[0][0], points[0][1]),
                     (points[i][0], points[i][1]),
                     col,
                     thickness)
        return img, points

    def gaussian_noise(self, img):
        """ Apply random noise to the image """
        cv2.randu(img, 0, 255)
        return np.empty((0, 2), dtype=np.int)


if __name__ == '__main__':
    SEED = 1984
    '''
    IMAGE_SIZE_INIT = [960, 1280]
    IMAGE_SIZE_OUT = [240, 320]
    '''
    IMAGE_SIZE_INIT = [480, 640]
    IMAGE_SIZE_OUT = [120, 160]
    BLUR = 11
    SPLIT_SIZES = {'training': 15, 'validation': 15}
    OUT_PATH = '/data/datasets/synth_glunet'

    synth_dataset = GeometricPrimitives(SEED)

    funcs = [synth_dataset.draw_ellipses, synth_dataset.draw_lines, synth_dataset.draw_polygon, synth_dataset.draw_star]

    for split, size in SPLIT_SIZES.items():
        im_dir = osp.join(OUT_PATH, "images", split)
        if not osp.isdir(im_dir):
            os.makedirs(im_dir)

        for i in tqdm(range(size), desc=split, leave=False):
            image = synth_dataset.generate_background(IMAGE_SIZE_INIT)
            func = random.choice(funcs)
            image, _ = func(image)

            image = cv2.GaussianBlur(image, (BLUR, BLUR), 0)
            image = cv2.resize(image,
                               tuple(IMAGE_SIZE_OUT[::-1]),
                               interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(osp.join(im_dir, '{}.png'.format(i)), image)
    '''
    # Pack into a tar file
    tar = tarfile.open(tar_path, mode='w:gz')
    tar.add(temp_dir, arcname=primitive)
    tar.close()
    shutil.rmtree(temp_dir)
    tf.logging.info('Tarfile dumped to {}.'.format(tar_path))
    '''

