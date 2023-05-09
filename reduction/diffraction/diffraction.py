# -*- coding: utf-8 -*-

"""
This module defines the DiffractionScan class, which assembles a set of detector
images (which might have been recorded at different scattering angles) into a
diffractogram. The basic idea is to iterate over all detector images,
calculating the 'true' two-theta angle for each pixel and sorting the
corresponding intensities into their corresponding two-theta bin in the
diffractogram individually.

This allows for super-sampling, i.e. obtaining a diffractogram where the
step-size of the two-theta axis is much smaller then the pixel size of the
detector.
"""

import re
import numpy as np
import scipy.interpolate as interp

import matplotlib.pyplot as plt

__author__ = 'Florian Otte'

class DiffractionScan():

    def __init__(self, images, fio_filename, parameters, pixel_size = 172e-3,
        chip_size_x = 195, chip_size_y = 487, super_sampling = 1):
        """
        The default values are for PILATUS 100K detectors and no super-sampling.
        """

        self.PIXEL_SIZE      = pixel_size
        self.CHIP_Y          = chip_size_y
        self.CHIP_X          = chip_size_x
        self.SUPER_SAMPLE    = super_sampling

        self.img = images

        # i0:   Incoming intensity
        # tt:   two-theta: two times the scattering angle, or: Angle between the
        #       incoming beam direction and the inverse detector normal (in deg)
        self.i0, self.tt = self.read_fio(fio_filename)
        self.p = parameters

    def read_fio(self, fio_path):
        with open(fio_path, 'r') as content_file:
            content = content_file.read()

        pattern = r'\s*([+-]*\d+\.*\d*[e0-9-+]*)\s' * 17
        matches = re.findall(pattern, content)

        i0 = np.empty(len(matches))
        tt = np.empty(len(matches))

        for index, match in enumerate(matches):
            i0[index] = match[3]
            tt[index] = match[0]

        return i0, tt

    def get_diffractogram(self):
        x0, x1 = self.p['pil_pixel_x0'], self.p['pil_pixel_x1']
        y0, y1 = self.p['pil_pixel_y0'], self.p['pil_pixel_y1']

        # Create two vectors which describe the pixel positions along x and y,
        # in NORMAL SPACE of the detector area. The origin is choosen to be the
        # coordinate of the pixel which in line with the direct beam (at 0° tt).
        # The detector distance has to be measured between the sample position
        # and this pixel. Units are mm.
        pixel_positions_x = np.arange(
            -1*self.p['pil_pixel_direct_beam_x']+1,
            self.CHIP_X - self.p['pil_pixel_direct_beam_x']+1
            ) * self.PIXEL_SIZE
        pixel_positions_y = np.arange(
            -1*self.p['pil_pixel_direct_beam_y']+1,
            self.CHIP_Y - self.p['pil_pixel_direct_beam_y']+1
            ) * self.PIXEL_SIZE

        # XX, YY = np.meshgrid(pixel_positions_x, pixel_positions_y)

        #
        phi = np.arctan(pixel_positions_y / self.p['pil_distance'])

        # Correction because the detector surface is flat
        phi[phi > 0] = phi[phi > 0] * phi[phi > 0]/np.tan(phi[phi > 0])
        phi[phi < 0] = phi[phi < 0] * phi[phi < 0]/np.tan(phi[phi < 0])
        phi = phi*180/np.pi # Convert to degree

        # Calculate how many bins (i.e. discrete two-theta values) the new
        # diffractogram will have

        p_angle = 180/np.pi*np.arctan(self.PIXEL_SIZE / self.p['pil_distance'])
        l = np.ptp(self.tt) / p_angle

        # If SUPER_SAMPLE is 1, the two-theta axis of the diffractogram will
        # have a step-size corresponding to the solid angle of 1x the pixel-size

        bins = int(round(l)) * self.SUPER_SAMPLE

        # Create a new diffractogram in which we will sort pixel intensities
        tt_angle_min = np.min(np.min(self.tt) + phi)
        tt_angle_max = np.max(np.max(self.tt) + phi)
        diffrgm_x = np.linspace(tt_angle_min, tt_angle_max, bins)
        diffrgm_y = np.zeros(bins)
        intensity_mask = np.zeros(bins) + 1e-9

        images = self.img / self.i0[:, None, None]

        for idx, image in enumerate(images[:, x0:x1+1, y0:y1+1]):

            # Calculate the 3D coordinates of the center pixel, taking the
            # sample position as origin (0, 0, 0). The units are mm.
            cpx = 0
            cpy = np.sin(self.tt[idx] * np.pi / 180) * self.p['pil_distance']
            cpz = np.cos(self.tt[idx] * np.pi / 180) * self.p['pil_distance']

            # Use the center coordinates to calculate the position of all
            # detector pixels. Units are mm.
            px = pixel_positions_x + cpx
            py = pixel_positions_y * np.cos(self.tt[idx] * np.pi / 180) + cpy
            pz = cpz - pixel_positions_y * np.sin(self.tt[idx] * np.pi / 180)

            XX, YY = np.meshgrid(px, py)
            ZZ = np.tile(pz, (px.shape[0], 1)).T

            pixel_source_distance = np.sqrt(XX**2+YY**2+ZZ**2)
            perp_to_beam = np.sqrt(XX**2+YY**2)

            # Calculate true two-theta values for each pixel
            TT = (np.arcsin(perp_to_beam / pixel_source_distance) * 180 / np.pi).T

            # if idx % 5 == 0:
            #     plt.figure("2 Theta per pixel")
            #     plt.clf()
            #     plt.pcolormesh(XX, YY, TT.T, shading = "nearest")
            #     plt.xlabel("Pixel position X in space (mm)")
            #     plt.ylabel("Pixel position Y in space (mm)")
            #     plt.colorbar()
            #
            #     plt.figure("Trajectory")
            #     plt.plot(cpz, cpy, 'r.')
            #     plt.xlabel("Center pixel Z distance to sample")
            #     plt.ylabel("Center pixel Y distance to sample")

            # Apply ROI
            TT = TT[x0:x1+1, y0:y1+1]



            # l = list(range(TT[y0:y1+1].shape[1]))

            for idr, (x, y) in enumerate(zip(TT, image)):
                # print(x.shape, y.shape)

                f = interp.interp1d(x, y, 'nearest', fill_value = 0, bounds_error = False)
                m = interp.interp1d(x, np.ones(len(x)), fill_value = 0, bounds_error = False)

                diffrgm_y += f(diffrgm_x)
                intensity_mask += m(diffrgm_x)

            # if idx % 5 == 0:
            #     plt.figure("Integrating ...")
            #     plt.plot(diffrgm_x, diffrgm_y)
            # plt.pause(0.01)
            # # plt.plot(intensity_mask)
            # plt.show()



            progress = int(idx / self.img.shape[0] * 50)
            fmt = 'Full analysis pending:  [{:<50}]'.format('='*progress)
            print(fmt, end = '\r')

        progress = 50
        fmt = 'Full analysis pending:  [{:<50}]'.format('='*progress)
        print(fmt)

        return diffrgm_x, diffrgm_y / intensity_mask

    # def get_diffractogram_fast(self):
    #     x0, x1 = self.p['pil_pixel_x0'], self.p['pil_pixel_x1']
    #     y0, y1 = self.p['pil_pixel_y0'], self.p['pil_pixel_y1']


    #     pixel_positions_x = np.arange(
    #         -1*self.p['pil_pixel_direct_beam_x']+1,
    #         self.CHIP_X - self.p['pil_pixel_direct_beam_x']+1
    #         ) * self.PIXEL_SIZE
    #     pixel_positions_y = np.arange(
    #         -1*self.p['pil_pixel_direct_beam_y']+1,
    #         self.CHIP_Y - self.p['pil_pixel_direct_beam_y']+1
    #         ) * self.PIXEL_SIZE

    #     # XX, YY = np.meshgrid(pixel_positions_x, pixel_positions_y)

    #     phi = np.arctan(pixel_positions_y / self.p['pil_distance'])
    #     # correction because of flat detector surface
    #     # phi[phi > 0] = phi[phi > 0] * phi[phi > 0]/np.tan(phi[phi > 0])
    #     # phi[phi < 0] = phi[phi < 0] * phi[phi < 0]/np.tan(phi[phi < 0])

    #     # Convert to degree
    #     phi = phi*180/np.pi

    #     l = np.ptp(self.tt) / (180/np.pi*np.arctan(self.PIXEL_SIZE / self.p['pil_distance']))
    #     bins = int(round(l)) * self.SUPER_SAMPLE
    #     min_a, max_a = np.min(np.min(self.tt) + phi),  np.max(np.max(self.tt) + phi)
    #     diffrgm_x, diffrgm_y = np.linspace(min_a, max_a, bins), np.zeros(bins)


    #     for idx, image in enumerate(self.img[:, x0:x1+1, y0:y1+1]):
    #         x = self.tt[idx] + phi[y0:y1+1]
    #         y = np.sum(image, axis = 0)
    #         f = interp.interp1d(x, y, 'nearest', fill_value = 0, bounds_error = False)
    #         diffrgm_y += f(diffrgm_x)


    #         progress = int(idx / self.img.shape[0] * 50)
    #         fmt = 'Fast analysis pending:  [{:<50}]'.format('='*progress)
    #         print(fmt, end = '\r')

    #     #
    #     #     if idx % 5 == 0:
    #     #         plt.plot(x, y)
    #     # plt.show()

    #     progress = 50
    #     fmt = 'Fast analysis pending:  [{:<50}]'.format('='*progress)
    #     print(fmt)
    #     return diffrgm_x, diffrgm_y
