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

package org.photonvision.vision.pipe.impl;

import java.util.List;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.LoadJNI.JNITypes;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.apriltag.AprilTagDetectorType;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.apriltag.NapaTagDetector;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.CVPipe;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;

public class AprilTagDetectionPipe
        extends CVPipe<
                CVMat, List<AprilTagDetection>, AprilTagDetectionPipe.AprilTagDetectionPipeParams> {
    private static final Logger logger =
            new Logger(AprilTagDetectionPipe.class, LogGroup.VisionModule);

    private AprilTagDetector m_detector = new AprilTagDetector();
    // Exists only while the pipeline uses NAPA
    private NapaTagDetector m_napaDetector;
    private boolean m_warnedNapaMissing;

    public AprilTagDetectionPipe() {
        super();

        m_detector.addFamily("tag16h5");
        m_detector.addFamily("tag36h11");
    }

    @Override
    protected List<AprilTagDetection> process(CVMat in) {
        if (in.getMat().empty()) {
            return List.of();
        }

        if (m_detector == null) {
            throw new RuntimeException("Apriltag detector was released!");
        }

        if (m_napaDetector != null) {
            return m_napaDetector.detect(in.getMat(), params.napaMinTagPx());
        }

        var ret = m_detector.detect(in.getMat());

        if (ret == null) {
            return List.of();
        }

        return List.of(ret);
    }

    @Override
    public void setParams(AprilTagDetectionPipeParams newParams) {
        if (this.params == null || !this.params.equals(newParams)) {
            m_detector.setConfig(newParams.detectorParams());
            m_detector.setQuadThresholdParameters(newParams.quadParams());

            m_detector.clearFamilies();
            m_detector.addFamily(newParams.family().getNativeName());

            boolean useNapa = newParams.detector() == AprilTagDetectorType.kNapa;
            if (useNapa && !LoadJNI.hasLoaded(JNITypes.NAPA)) {
                if (!m_warnedNapaMissing) {
                    logger.warn("NAPA's library isn't loaded, so this pipeline uses AprilTag's detector");
                    m_warnedNapaMissing = true;
                }
                useNapa = false;
            }
            if (useNapa) {
                String family = newParams.family().getNativeName();
                // A new family needs a new detector, created before the old one is closed. Creating it
                // fails only when NAPA is out of memory; this then throws on every frame until the
                // settings change, and the old detector stays valid.
                var napaDetector =
                        m_napaDetector != null && m_napaDetector.getFamily().equals(family)
                                ? m_napaDetector
                                : new NapaTagDetector(family);
                napaDetector.setThreads(newParams.detectorParams().numThreads);
                napaDetector.setCalibration(newParams.calibration());
                if (napaDetector != m_napaDetector) {
                    if (m_napaDetector != null) {
                        m_napaDetector.close();
                    }
                    m_napaDetector = napaDetector;
                }
            } else if (m_napaDetector != null) {
                // Free NAPA's worker threads and buffers while AprilTag runs
                m_napaDetector.close();
                m_napaDetector = null;
            }
        }

        super.setParams(newParams);
    }

    @Override
    public void release() {
        m_detector.close();
        m_detector = null;
        if (m_napaDetector != null) {
            m_napaDetector.close();
            m_napaDetector = null;
        }
    }

    /**
     * @param detectorParams AprilTag's settings; NAPA uses only their thread count
     * @param detector the detector to run
     * @param napaMinTagPx the smallest tag side, in pixels, NAPA is guaranteed to find
     * @param calibration the camera's calibration at the current resolution, or null; NAPA fits tag
     *     sides through its lens. Compared by reference: FrameStaticProperties keeps the same
     *     instance from frame to frame.
     */
    public static record AprilTagDetectionPipeParams(
            AprilTagFamily family,
            AprilTagDetector.Config detectorParams,
            AprilTagDetector.QuadThresholdParameters quadParams,
            AprilTagDetectorType detector,
            int napaMinTagPx,
            CameraCalibrationCoefficients calibration) {}
}
