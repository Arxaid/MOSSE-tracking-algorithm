# This file is part of the MOSSE Tracking Algorithm project
#
# Copyright (c) 2018 Tianhong Dai
# Copyright (c) 2023 Vladislav Sosedov

import numpy as np
import cv2
import os
from datetime import datetime
from src.utils import linear_mapping, pre_process, random_warp

class mosse:
    """
    Implementation of the basic MOSSE (Minimum Output Sum of Squared Error)\n
    tracking algorithm for real-time image processing from the webcam.
    """
    def __init__(self, args):
        """ MOSSE tracking algorithm constructor.
        #### Parameters:
        args: Namespace:
        1. lr: float -- Learning rate (default = 0.125);
        2. sigma: float -- Sigma value (default = 100);
        3. num_pretrain: int -- Number of pretrain (default = 128);
        4. rotate: bool -- Rotate image during pre-training;
        5. mode: bool -- ROI search mode. 0 - manual, 1 - correlation search algorithm.
        """
        self.args = args
        # Check your webcam index first:
        # ls /dev/ | grep video
        self.cam = cv2.VideoCapture(0)
    
    def tracking(self):
        """ MOSSE tracking algorithm, including:\n
        1. ROI initialization;
        2. Filter pre-training;
        3. Tracking process and its visualization.
        """
        frame_idx = int(0)
        startTimestamp = datetime.now().timestamp()

        cv2.namedWindow('MOSSE tracking algorithm ')
        init_ret, init_img = self.cam.read()
        assert init_ret, 'Failed to capture image from current webcam.'

        init_frame = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)

        if self.args.mode == 0:
            init_gt = cv2.selectROI('MOSSE tracking algorithm ', init_img, False, False)
        if self.args.mode == 1:
            reference_image = cv2.imread('reference/target_mark.png')
            heatmap = cv2.matchTemplate(init_img, reference_image, cv2.TM_CCOEFF_NORMED)
            h, w = reference_image.shape[:-1]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            init_gt = (x, y, x+w, y+h)
        
        init_gt = np.array(init_gt).astype(np.int64)

        response_map = self._get_gauss_response(init_frame, init_gt)

        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        G = np.fft.fft2(g)

        Ai, Bi = self._pre_training(fi, G)

        while True:
            current_ret, current_frame = self.cam.read()
            assert current_ret, 'Failed to capture image from current webcam.'
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)

            if frame_idx == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                pos = init_gt.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                Gi = Hi * np.fft.fft2(fi)
                gi = linear_mapping(np.fft.ifft2(Gi))

                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
                
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0]+pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1]+pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))

                Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Ai
                Bi = self.args.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.args.lr) * Bi
            
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            cv2.imshow('MOSSE tracking algorithm ', current_frame)
            frame_idx += 1
            k = cv2.waitKey(1)
            if k%256 == 27:
                endTimestamp = datetime.now().timestamp()
                print(f'FPS: {(endTimestamp - startTimestamp)/frame_idx * 1000}')
                break

    def _pre_training(self, init_frame, G):
        """ Filter pre-training.
        #### Parameters:
        1. init_frame: NDArray[float32] -- Initial image for which the ROI is selected.
        2. G: NDArray[complex128] -- Initial image's Fourier domain.
        #### Returns:
        1. Ai -- MOSSE filter's numerator.
        2. Bi -- MOSSE filter's divider.
        """
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.args.num_pretrain):
            if self.args.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi

    def _get_gauss_response(self, img, gt):
        """ Ground-truth gaussian reponse.
        #### Parameters:
        1. img: NDArray[float32] -- Image for which the ROI is selected.
        2. gt: NDArray[float64] -- ROI.
        #### Returns:
        resopnse -- Ground-truth gaussian reponse.
        """
        height, width = img.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        response = np.exp(-dist)
        response = linear_mapping(response)
        return response