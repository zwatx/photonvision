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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigProvider;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagDetectorType;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.apriltag.NapaTagDetector;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.vision.apriltag.AprilTagDetection;

/** NAPA, run through the AprilTag pipeline, against WPILib's AprilTagDetector. */
public class NapaAprilTagTest {
    private record Tag(int id, List<Point> corners, Point center, Transform3d pose) {}

    // Each run loads a fresh calibration
    private static final Supplier<CameraCalibrationCoefficients> kLifeCam =
            () -> TestUtils.get2020LifeCamCoeffs(false);
    private static final Supplier<CameraCalibrationCoefficients> kLimelight =
            () -> TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false);

    @BeforeEach
    public void setup() throws IOException {
        LoadJNI.loadLibraries();
        LoadJNI.forceLoad(LoadJNI.JNITypes.NAPA);
        ConfigProvider.getInstance().load();
    }

    static Stream<Arguments> images() {
        return Stream.of(
                Arguments.of(
                        TestUtils.ApriltagTestImages.kTag1_640_480, AprilTagFamily.kTag36h11, kLifeCam),
                Arguments.of(
                        TestUtils.ApriltagTestImages.kTag1_16h5_1280, AprilTagFamily.kTag16h5, kLimelight),
                Arguments.of(
                        TestUtils.ApriltagTestImages.kTag_corner_1280, AprilTagFamily.kTag16h5, kLimelight),
                Arguments.of(
                        TestUtils.ApriltagTestImages.k36h11_stress_test, AprilTagFamily.kTag36h11, kLimelight));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("images")
    public void testNapaMatchesAprilTag(
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
        // Summed corner offsets: a slip in the pixel convention moves every corner the same way
        double offsetX = 0;
        double offsetY = 0;
        int compared = 0;
        for (Tag expected : aprilTag) {
            // Images can repeat an ID, so match each tag to NAPA's nearest with the same ID, within
            // 5 px of AprilTag's center
            Tag actual = null;
            double nearest = 5;
            for (Tag candidate : napa) {
                double distance = distance(expected.center(), candidate.center());
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
                Point a = expected.corners().get(k);
                Point b = actual.corners().get(k);
                worstCornerPx = Math.max(worstCornerPx, distance(a, b));
                offsetX += b.x - a.x;
                offsetY += b.y - a.y;
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
                        + " px, mean offset ("
                        + offsetX / (4 * compared)
                        + ", "
                        + offsetY / (4 * compared)
                        + ") px");
        assertTrue(compared > 0, "no tags to compare in " + image);
        assertTrue(worstCornerPx < 2, "corners differ from AprilTag's by " + worstCornerPx + " px");
        assertTrue(
                Math.hypot(offsetX, offsetY) / (4 * compared) < 0.25,
                "corners are offset from AprilTag's on average");
    }

    private static double distance(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    private static List<Tag> detect(
            TestUtils.ApriltagTestImages image,
            AprilTagFamily family,
            Supplier<CameraCalibrationCoefficients> calibration,
            AprilTagDetectorType detector) {
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().solvePNPEnabled = true;
            pipeline.getSettings().outputMaximumTargets = 1000;
            return run(pipeline, image, family, calibration, detector);
        }
    }

    private static List<Tag> run(
            AprilTagPipeline pipeline,
            TestUtils.ApriltagTestImages image,
            AprilTagFamily family,
            Supplier<CameraCalibrationCoefficients> calibration,
            AprilTagDetectorType detector) {
        pipeline.getSettings().tagFamily = family;
        pipeline.getSettings().detector = detector;
        List<Tag> tags = new ArrayList<>();
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
        return tags;
    }

    @Test
    public void testSwitchingDetectorsAndFamilies() {
        // One pipeline switching detector and family gives what a fresh pipeline gives for each
        // setting: NAPA, AprilTag, NAPA again, then NAPA with another family.
        var tag36h11 = TestUtils.ApriltagTestImages.kTag1_640_480;
        var tag16h5 = TestUtils.ApriltagTestImages.kTag1_16h5_1280;
        var napa = AprilTagDetectorType.kNapa;
        var aprilTag = AprilTagDetectorType.kAprilTag;
        try (var pipeline = new AprilTagPipeline()) {
            pipeline.getSettings().solvePNPEnabled = true;
            pipeline.getSettings().outputMaximumTargets = 1000;
            for (var detector : List.of(napa, aprilTag, napa)) {
                List<Tag> tags = run(pipeline, tag36h11, AprilTagFamily.kTag36h11, kLifeCam, detector);
                assertFalse(tags.isEmpty(), detector + " found no tags");
                assertEquals(detect(tag36h11, AprilTagFamily.kTag36h11, kLifeCam, detector), tags);
            }
            List<Tag> tags = run(pipeline, tag16h5, AprilTagFamily.kTag16h5, kLimelight, napa);
            assertFalse(tags.isEmpty(), "NAPA found no tag16h5 tags");
            assertEquals(detect(tag16h5, AprilTagFamily.kTag16h5, kLimelight, napa), tags);
        }
    }

    @Test
    public void testHomographyMapsUpstreamTagCoordinates() {
        // The homography maps upstream's tag coordinates onto the corners, and its last column is
        // the tag's center.
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
        assertEquals(22, loaded.napaMinTagPx);
    }

    private static Mat grayImage(TestUtils.ApriltagTestImages image) {
        return Imgcodecs.imread(
                TestUtils.getApriltagImagePath(image, false).toString(), Imgcodecs.IMREAD_GRAYSCALE);
    }

    private static List<String> summary(List<AprilTagDetection> detections) {
        return detections.stream()
                .map(
                        d ->
                                d.getId()
                                        + " "
                                        + d.getHamming()
                                        + " "
                                        + d.getDecisionMargin()
                                        + " "
                                        + Arrays.toString(d.getCorners()))
                .toList();
    }

    @Test
    public void testThreadsFindTheSameTags() {
        Mat image = grayImage(TestUtils.ApriltagTestImages.k36h11_stress_test);
        try (var serial = new NapaTagDetector("tag36h11");
                var threaded = new NapaTagDetector("tag36h11")) {
            threaded.setThreads(4);
            List<String> expected = summary(serial.detect(image, 16));
            assertTrue(expected.size() > 100);
            assertEquals(expected, summary(threaded.detect(image, 16)));
            assertEquals(expected, summary(threaded.detect(image, 16))); // reused worker buffers
        } finally {
            image.release();
        }
    }

    private static CameraCalibrationCoefficients withLensModel(
            CameraCalibrationCoefficients calibration, CameraLensModel model) {
        return new CameraCalibrationCoefficients(
                calibration.resolution,
                calibration.cameraIntrinsics,
                calibration.distCoeffs,
                calibration.calobjectWarp,
                calibration.observations,
                calibration.calobjectSize,
                calibration.calobjectSpacing,
                model,
                calibration.optimizationInputs);
    }

    private static double largestCornerMove(AprilTagDetection a, AprilTagDetection b) {
        double[] p = a.getCorners();
        double[] q = b.getCorners();
        double largest = 0;
        for (int k = 0; k < 8; k += 2) {
            largest = Math.max(largest, Math.hypot(p[k] - q[k], p[k + 1] - q[k + 1]));
        }
        return largest;
    }

    @Test
    public void testLensAppliesAtTheCalibrationsResolution() {
        Mat image = grayImage(TestUtils.ApriltagTestImages.kTag_corner_1280);
        Mat cropped = image.submat(0, 700, 0, 1270);
        // Saved without a lens model, as calibrations were before mrcal: OpenCV's
        var calibration = TestUtils.getCoeffs(TestUtils.LIMELIGHT_480P_CAL_FILE, false);
        try (var detector = new NapaTagDetector("tag16h5")) {
            List<AprilTagDetection> plain = detector.detect(image, 16);
            assertEquals(1, plain.size());

            // This mild lens barely bends so small a tag (about 0.005 px), but it does move its corners
            detector.setCalibration(calibration);
            List<AprilTagDetection> throughLens = detector.detect(image, 16);
            assertTrue(detector.isLensApplied());
            assertEquals(plain.get(0).getId(), throughLens.get(0).getId());
            double moved = largestCornerMove(plain.get(0), throughLens.get(0));
            assertTrue(moved > 0 && moved < 1, "the lens moved a corner " + moved + " px");

            // Another resolution fits in pixels; the calibrated one uses the lens again
            detector.detect(cropped, 16);
            assertFalse(detector.isLensApplied());
            assertEquals(summary(throughLens), summary(detector.detect(image, 16)));
            assertTrue(detector.isLensApplied());

            // Clearing the calibration, or giving an mrcal model, while the lens is in use: exactly
            // the results without one
            detector.setCalibration(null);
            assertEquals(summary(plain), summary(detector.detect(image, 16)));
            assertFalse(detector.isLensApplied());
            detector.setCalibration(withLensModel(calibration, CameraLensModel.LENSMODEL_OPENCV));
            assertEquals(summary(throughLens), summary(detector.detect(image, 16)));
            detector.setCalibration(withLensModel(calibration, CameraLensModel.LENSMODEL_STERIOGRAPHIC));
            assertEquals(summary(plain), summary(detector.detect(image, 16)));
            assertFalse(detector.isLensApplied());
        } finally {
            cropped.release();
            image.release();
            calibration.release(); // the copies share its matrices
        }
    }

    @Test
    public void testBlankAndTinyImages() {
        var blank = new Mat(480, 640, CvType.CV_8UC1, new Scalar(128));
        var tiny = new Mat(8, 8, CvType.CV_8UC1, new Scalar(0));
        var color = new Mat(480, 640, CvType.CV_8UC3, new Scalar(0, 0, 0));
        try (var detector = new NapaTagDetector("tag36h11")) {
            assertTrue(detector.detect(blank, 32).isEmpty());
            assertTrue(detector.detect(tiny, 32).isEmpty());
            assertThrows(IllegalArgumentException.class, () -> detector.detect(color, 32));
        } finally {
            color.release();
            tiny.release();
            blank.release();
        }
        assertThrows(IllegalArgumentException.class, () -> new NapaTagDetector("tag99h99"));
        assertThrows(IllegalArgumentException.class, () -> new NapaTagDetector(null));
    }

    @Test
    public void testImageViewFindsTheSameTagsAsACopy() {
        Mat image = grayImage(TestUtils.ApriltagTestImages.k36h11_stress_test);
        // Rows of a view are further apart than its width
        Mat view = image.submat(1, image.rows() - 1, 3, image.cols() - 5);
        Mat copy = view.clone();
        try (var detector = new NapaTagDetector("tag36h11")) {
            assertFalse(view.isContinuous());
            List<String> expected = summary(detector.detect(copy, 16));
            assertTrue(expected.size() > 100);
            assertEquals(expected, summary(detector.detect(view, 16)));
        } finally {
            copy.release();
            view.release();
            image.release();
        }
    }
}
