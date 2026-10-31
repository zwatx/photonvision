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
import org.photonvision.jni.NapaDetectorJni;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * NAPA's AprilTag detector, reporting tags as WPILib's AprilTagDetector does: corners in upstream
 * AprilTag's order with (0, 0) at the top-left pixel's corner, and the homography from upstream's
 * tag coordinates.
 */
public class NapaTagDetector implements AutoCloseable {
    /** The smallest and largest values NAPA accepts for the minimum tag size. */
    public static final int kMinTagPxLimit = 8;

    public static final int kMaxTagPxLimit = 30000;

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

    /**
     * @param family a tag family's native name, such as "tag36h11"
     */
    public NapaTagDetector(String family) {
        m_handle = NapaDetectorJni.create(family);
        if (m_handle == 0) {
            throw new IllegalArgumentException("NAPA doesn't support the tag family " + family);
        }
        m_family = family;
        m_maxHamming = NapaDetectorJni.recommendedHamming(family);
    }

    public String getFamily() {
        return m_family;
    }

    /**
     * Detects tags in an 8-bit grayscale image.
     *
     * @param minTagPx the smallest tag side, in pixels, NAPA is guaranteed to find
     */
    public List<AprilTagDetection> detect(Mat image, int minTagPx) {
        if (m_handle == 0) {
            throw new IllegalStateException("NAPA detector was closed");
        }
        if (image.type() != CvType.CV_8UC1) {
            throw new IllegalArgumentException("NAPA needs an 8-bit grayscale image");
        }
        if (image.cols() < 16 || image.rows() < 16) {
            return List.of();
        }

        double[] values =
                NapaDetectorJni.detect(
                        m_handle,
                        image.dataAddr(),
                        image.cols(),
                        image.rows(),
                        (int) image.step1(0),
                        Math.clamp(minTagPx, kMinTagPxLimit, kMaxTagPxLimit),
                        m_maxHamming,
                        kMinContrast,
                        kMinBorderCells,
                        kSharpeningX100);
        if (values == null) {
            return List.of();
        }

        int count = values.length / NapaDetectorJni.kValuesPerDetection;
        List<AprilTagDetection> detections = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int offset = i * NapaDetectorJni.kValuesPerDetection;
            double[] corners = new double[8];
            for (int k = 0; k < 4; k++) {
                int napaCorner = offset + 3 + 2 * kUpstreamOrder[k];
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
                            homography[2],
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

    @Override
    public void close() {
        if (m_handle != 0) {
            NapaDetectorJni.destroy(m_handle);
            m_handle = 0;
        }
    }
}
