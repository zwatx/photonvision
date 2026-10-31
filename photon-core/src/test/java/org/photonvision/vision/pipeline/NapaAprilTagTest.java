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

package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.avaje.jsonb.Jsonb;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigProvider;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagDetectorType;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.apriltag.NapaTagDetector;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.wpilib.math.geometry.Transform3d;

/** NAPA, run through the AprilTag pipeline, against WPILib's AprilTagDetector. */
public class NapaAprilTagTest {
    private record Tag(int id, List<Point> corners, Point centre, Transform3d pose) {}

    @BeforeEach
    public void setup() {
        LoadJNI.loadLibraries();
        ConfigProvider.getInstance().load();
    }

    static Stream<Arguments> images() {
        Supplier<CameraCalibrationCoefficients> lifeCam = () -> TestUtils.get2020LifeCamCoeffs(false);
        Supplier<CameraCalibrationCoefficients> limelight =
                () -> TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false);
        return Stream.of(
                Arguments.of(TestUtils.ApriltagTestImages.kTag1_640_480, AprilTagFamily.kTag36h11, lifeCam),
                Arguments.of(
                        TestUtils.ApriltagTestImages.kTag1_16h5_1280, AprilTagFamily.kTag16h5, limelight),
                Arguments.of(
                        TestUtils.ApriltagTestImages.kTag_corner_1280, AprilTagFamily.kTag16h5, limelight),
                Arguments.of(
                        TestUtils.ApriltagTestImages.k36h11_stress_test, AprilTagFamily.kTag36h11, limelight));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("images")
    public void testNapaFindsAprilTagsTags(
            TestUtils.ApriltagTestImages image,
            AprilTagFamily family,
            Supplier<CameraCalibrationCoefficients> calibration) {
        List<Tag> aprilTag = detect(image, family, calibration, AprilTagDetectorType.kAprilTag);
        List<Tag> napa = detect(image, family, calibration, AprilTagDetectorType.kNapa);
        assertFalse(aprilTag.isEmpty(), "AprilTag found no tags in " + image);

        // The pipeline reports at most Packet.MAX_ARRAY_LEN targets, so when both hit that cap each
        // keeps a different subset: compare the tags they share.
        boolean capped = aprilTag.size() == Packet.MAX_ARRAY_LEN;
        if (capped) {
            assertEquals(Packet.MAX_ARRAY_LEN, napa.size());
        }

        double worstCornerPx = 0;
        int compared = 0;
        for (Tag expected : aprilTag) {
            // Images can repeat an ID, so match each tag to NAPA's nearest with the same ID
            Tag actual = null;
            double nearest = 5;
            for (Tag candidate : napa) {
                double distance = distance(expected.centre(), candidate.centre());
                if (candidate.id() == expected.id() && distance < nearest) {
                    actual = candidate;
                    nearest = distance;
                }
            }
            if (capped && actual == null) {
                continue;
            }
            assertNotNull(actual, "NAPA missed tag " + expected.id() + " in " + image);
            compared++;
            for (int k = 0; k < 4; k++) {
                worstCornerPx =
                        Math.max(worstCornerPx, distance(expected.corners().get(k), actual.corners().get(k)));
            }
            double range = expected.pose().getTranslation().getNorm();
            double offset = expected.pose().getTranslation().getDistance(actual.pose().getTranslation());
            assertTrue(
                    offset < 0.05 * range,
                    "tag " + expected.id() + " pose is " + offset + " m from AprilTag's at " + range + " m");
        }
        System.out.println(
                image
                        + ": AprilTag "
                        + aprilTag.size()
                        + " tags, NAPA "
                        + napa.size()
                        + ", "
                        + compared
                        + " compared, worst corner difference "
                        + worstCornerPx
                        + " px");
        assertTrue(compared > 0, "no tags to compare in " + image);
        assertTrue(worstCornerPx < 2, "corners differ from AprilTag's by " + worstCornerPx + " px");
    }

    private static double distance(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    private static List<Tag> detect(
            TestUtils.ApriltagTestImages image,
            AprilTagFamily family,
            Supplier<CameraCalibrationCoefficients> calibration,
            AprilTagDetectorType detector) {
        List<Tag> tags = new ArrayList<>();
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().solvePNPEnabled = true;
            pipeline.getSettings().tagFamily = family;
            pipeline.getSettings().detector = detector;
            pipeline.getSettings().outputMaximumTargets = 1000;

            try (var frameProvider =
                    new FileFrameProvider(
                            TestUtils.getApriltagImagePath(image, false),
                            TestUtils.WPI2020Image.FOV,
                            calibration.get())) {
                frameProvider.requestFrameThresholdType(pipeline.getThresholdType());
                try (CVPipelineResult result =
                        pipeline.run(frameProvider.get(), QuirkyCamera.DefaultCamera)) {
                    for (var target : result.targets) {
                        var corners = List.copyOf(target.getTargetCorners());
                        double x = 0;
                        double y = 0;
                        for (Point corner : corners) {
                            x += corner.x / 4;
                            y += corner.y / 4;
                        }
                        tags.add(
                                new Tag(
                                        target.getFiducialId(),
                                        corners,
                                        new Point(x, y),
                                        target.getBestCameraToTarget3d()));
                    }
                }
            }
        }
        return tags;
    }

    @Test
    public void testDetectorReportsUpstreamConventions() {
        // The homography maps upstream's tag coordinates onto the corners, and its last column is
        // the tag's centre.
        double[] corners = {100.5, 300.5, 300.5, 300.5, 300.5, 100.5, 100.5, 100.5};
        double[] h = NapaTagDetector.homography(corners);
        double[][] tagPoints = {{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
        for (int k = 0; k < 4; k++) {
            double u = tagPoints[k][0];
            double v = tagPoints[k][1];
            double w = h[6] * u + h[7] * v + h[8];
            assertEquals(corners[2 * k], (h[0] * u + h[1] * v + h[2]) / w, 1e-9);
            assertEquals(corners[2 * k + 1], (h[3] * u + h[4] * v + h[5]) / w, 1e-9);
        }
        assertEquals(200.5, h[2], 1e-9);
        assertEquals(200.5, h[5], 1e-9);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testUpstreamPipelinesLoadWithNapa() {
        var settings = new AprilTagPipelineSettings();
        settings.detector = AprilTagDetectorType.kAprilTag;
        var json = Jsonb.instance().type(CVPipelineSettings.class).toJson(settings);
        // A pipeline saved by upstream PhotonVision has neither field
        var fields = (Map<String, Object>) Jsonb.instance().type(Object.class).fromJson(json);
        assertNotNull(fields.remove("detector"));
        assertNotNull(fields.remove("napaMinTagPx"));
        var upstreamJson = Jsonb.instance().type(Object.class).toJson(fields);

        var loaded =
                (AprilTagPipelineSettings)
                        Jsonb.instance().type(CVPipelineSettings.class).fromJson(upstreamJson);
        assertEquals(AprilTagDetectorType.kNapa, loaded.detector);
        assertEquals(16, loaded.napaMinTagPx);
    }

    @Test
    public void testBlankAndTinyImages() {
        try (var detector = new NapaTagDetector("tag36h11")) {
            var blank = new Mat(480, 640, CvType.CV_8UC1, new Scalar(128));
            assertTrue(detector.detect(blank, 32).isEmpty());
            var tiny = new Mat(8, 8, CvType.CV_8UC1, new Scalar(0));
            assertTrue(detector.detect(tiny, 32).isEmpty());
            var color = new Mat(480, 640, CvType.CV_8UC3, new Scalar(0, 0, 0));
            assertThrows(IllegalArgumentException.class, () -> detector.detect(color, 32));
            Imgproc.rectangle(blank, new Point(10, 10), new Point(20, 20), new Scalar(0));
            assertTrue(detector.detect(blank.submat(5, 400, 5, 600), 32).isEmpty());
        }
        assertThrows(IllegalArgumentException.class, () -> new NapaTagDetector("tag99h99"));
    }
}
