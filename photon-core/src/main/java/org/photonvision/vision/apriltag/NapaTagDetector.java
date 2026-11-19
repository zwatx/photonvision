/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.apriltag;

import java.util.ArrayList;
import java.util.List;
import org.ejml.data.SingularMatrixException;
import org.ejml.simple.SimpleMatrix;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.napa.NapaJNI;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * NAPA's AprilTag detector, reporting tags as WPILib's AprilTagDetector does: corners in upstream
 * AprilTag's order with (0, 0) at the top-left pixel's corner, and the homography from upstream's
 * tag coordinates.
 */
public class NapaTagDetector implements AutoCloseable {
    private static final Logger logger = new Logger(NapaTagDetector.class, LogGroup.VisionModule);

    // The ranges napa_core.h accepts. The UI's sliders stay well inside them.
    private static final int kSmallestMinTagPx = 8;
    private static final int kLargestMinTagPx = 30000;
    private static final int kMaxThreads = 16;

    // NAPA's defaults, as napa_vision.NapaDetector uses them
    private static final int kMinContrast = 20;
    private static final int kMinBorderCells = 22;
    private static final int kSharpeningX100 = 25;

    // NAPA's corner order is upstream AprilTag's p[1], p[0], p[3], p[2]
    private static final int[] kUpstreamOrder = {1, 0, 3, 2};
    // Upstream AprilTag's tag coordinates for its corners p[0..3]
    private static final double[][] kTagPoints = {{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};

    private final String m_family;
    private final int m_maxHamming;
    private long m_handle;
    private int m_threads = 1;
    // The last calibration passed to setCalibration, so repeated calls cost nothing
    private CameraCalibrationCoefficients m_requestedCalibration;
    // The calibration whose lens NAPA uses at that calibration's resolution: null without one, for an
    // mrcal model, or once NAPA has rejected it
    private CameraCalibrationCoefficients m_lensCalibration;
    // Whether NAPA's lens is set now, for the last image's resolution
    private boolean m_lensApplied;

    /**
     * @param family a tag family's native name, such as "tag36h11"
     */
    public NapaTagDetector(String family) {
        // Before creating the detector, so a failure here can't leak it
        m_maxHamming = NapaJNI.recommendedHamming(family);
        m_handle = m_maxHamming < 0 ? 0 : NapaJNI.create(family);
        if (m_handle == 0) {
            throw new IllegalArgumentException(
                    "NAPA couldn't create a detector for the tag family " + family);
        }
        m_family = family;
    }

    public String getFamily() {
        return m_family;
    }

    /**
     * Sets how many threads each detection may use. Results are the same at every count; only latency
     * changes. If the threads can't be started, NAPA keeps the previous count.
     *
     * @param threads clamped to 1 through 16
     */
    public void setThreads(int threads) {
        requireOpen();
        threads = Math.clamp(threads, 1, kMaxThreads);
        if (threads == m_threads) {
            return;
        }
        if (NapaJNI.setThreads(m_handle, threads) == 0) {
            m_threads = threads;
        } else {
            logger.warn("NAPA couldn't start " + threads + " threads; it keeps using " + m_threads);
        }
    }

    /**
     * Fits tag sides through the camera's lens, which straightens the corners of tags the lens bends.
     * The lens applies to images at the calibration's resolution; others, a null calibration and
     * mrcal models fit sides in pixels.
     */
    public void setCalibration(CameraCalibrationCoefficients calibration) {
        requireOpen();
        if (calibration == m_requestedCalibration) {
            return;
        }
        m_requestedCalibration = calibration;
        m_lensCalibration = calibration != null && isOpenCvModel(calibration) ? calibration : null;
        if (m_lensApplied) {
            NapaJNI.setLens(m_handle, NapaJNI.kLensNone, null, null);
            m_lensApplied = false;
        }
    }

    /** Whether NAPA's lens is set, for the last image's resolution. */
    public boolean isLensApplied() {
        return m_lensApplied;
    }

    // Calibrations saved before lens models were recorded have none: they are OpenCV's, as the pose
    // estimators assume.
    private static boolean isOpenCvModel(CameraCalibrationCoefficients calibration) {
        return calibration.lensmodel == null
                || calibration.lensmodel == CameraLensModel.LENSMODEL_OPENCV;
    }

    // Sets or clears NAPA's lens for an image: a calibration holds only at its own resolution.
    private void applyLens(Mat image) {
        boolean wanted =
                m_lensCalibration != null
                        && m_lensCalibration.resolution.width == image.cols()
                        && m_lensCalibration.resolution.height == image.rows();
        if (wanted == m_lensApplied) {
            return;
        }
        if (wanted) {
            double[] k = m_lensCalibration.getIntrinsicsArr(); // the camera matrix, row by row
            m_lensApplied =
                    k.length == 9
                            && NapaJNI.setLens(
                                            m_handle,
                                            NapaJNI.kLensBrownConrady,
                                            new double[] {k[0], k[4], k[2], k[5], k[1]},
                                            m_lensCalibration.getDistCoeffsArr())
                                    == 0;
            if (!m_lensApplied) {
                logger.warn(
                        "NAPA can't use this camera calibration's lens, so it fits tags in pixels: "
                                + k.length
                                + " camera matrix values, "
                                + m_lensCalibration.getDistCoeffsArr().length
                                + " distortion coefficients");
                m_lensCalibration = null; // until the calibration changes
            }
        } else {
            NapaJNI.setLens(m_handle, NapaJNI.kLensNone, null, null);
            m_lensApplied = false;
        }
    }

    /**
     * Detects tags in an 8-bit grayscale image.
     *
     * @param minTagPx the smallest tag side, in pixels, NAPA is guaranteed to find
     * @throws IllegalArgumentException if the image isn't 8-bit grayscale
     * @throws IllegalStateException if the detector is closed, or NAPA fails on the image, such as
     *     when out of memory
     */
    public List<AprilTagDetection> detect(Mat image, int minTagPx) {
        requireOpen();
        if (image.type() != CvType.CV_8UC1) {
            throw new IllegalArgumentException("NAPA needs an 8-bit grayscale image");
        }
        applyLens(image);

        double[] values =
                NapaJNI.detect(
                        m_handle,
                        image.dataAddr(),
                        image.cols(),
                        image.rows(),
                        (int) image.step1(0),
                        Math.clamp(minTagPx, kSmallestMinTagPx, kLargestMinTagPx),
                        m_maxHamming,
                        kMinContrast,
                        kMinBorderCells,
                        kSharpeningX100);
        if (values == null) {
            throw new IllegalStateException(
                    "NAPA failed on a " + image.cols() + "x" + image.rows() + " image");
        }

        int count = values.length / NapaJNI.kValuesPerDetection;
        List<AprilTagDetection> detections = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int offset = i * NapaJNI.kValuesPerDetection;
            double[] corners = new double[8];
            for (int k = 0; k < 4; k++) {
                int napaCorner = offset + 3 + 2 * kUpstreamOrder[k]; // corners follow ID, hamming, margin
                corners[2 * k] = values[napaCorner] + 0.5;
                corners[2 * k + 1] = values[napaCorner + 1] + 0.5;
            }
            double[] homography;
            try {
                homography = homography(corners);
            } catch (SingularMatrixException e) {
                continue; // the corners don't form a quadrilateral
            }
            detections.add(
                    new AprilTagDetection(
                            m_family,
                            (int) values[offset],
                            (int) values[offset + 1],
                            (float) values[offset + 2],
                            homography,
                            homography[2], // the tag's center, where the homography maps (0, 0)
                            homography[5],
                            corners));
        }
        return detections;
    }

    /**
     * The homography from upstream AprilTag's tag coordinates to these image corners, row-major, with
     * its last entry 1.
     */
    public static double[] homography(double[] corners) {
        var system = new SimpleMatrix(8, 8);
        var image = new SimpleMatrix(8, 1);
        for (int k = 0; k < 4; k++) {
            double u = kTagPoints[k][0];
            double v = kTagPoints[k][1];
            double x = corners[2 * k];
            double y = corners[2 * k + 1];
            int row = 2 * k;
            system.setRow(row, 0, u, v, 1, 0, 0, 0, -u * x, -v * x);
            system.setRow(row + 1, 0, 0, 0, 0, u, v, 1, -u * y, -v * y);
            image.set(row, 0, x);
            image.set(row + 1, 0, y);
        }
        var solution = system.solve(image);
        double[] homography = new double[9];
        for (int i = 0; i < 8; i++) {
            homography[i] = solution.get(i, 0);
        }
        homography[8] = 1;
        return homography;
    }

    private void requireOpen() {
        if (m_handle == 0) {
            throw new IllegalStateException("NAPA detector was closed");
        }
    }

    @Override
    public void close() {
        if (m_handle != 0) {
            NapaJNI.destroy(m_handle);
            m_handle = 0;
        }
    }
}
