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

package org.photonvision.jni;

/**
 * Bindings for NAPA's AprilTag detector (https://github.com/zwatx/NAPA), compiled into
 * photontargetingJNI. A handle is not thread-safe; use one per camera.
 */
public class NapaDetectorJni {
    /** Values per tag in {@link #detect}'s result: ID, hamming, decision margin, 8 corner values. */
    public static final int kValuesPerDetection = 11;

    /**
     * Creates a detector for one tag family, such as "tag36h11" or "tag16h5".
     *
     * @return a handle, or 0 if the family is unknown
     */
    public static native long create(String family);

    /**
     * The number of payload bits NAPA corrects by default for a family: as unlikely to accept a
     * chance match as tag36h11 with 3.
     *
     * @return the bit count, or -1 if the family is unknown
     */
    public static native int recommendedHamming(String family);

    public static native void destroy(long handle);

    /**
     * Detects tags in an 8-bit grayscale image.
     *
     * @param dataAddr the address of the image's first pixel
     * @param stride bytes from one row to the next
     * @return {@link #kValuesPerDetection} values per tag, its corners as x, y pairs in OpenCV's
     *     order with (0, 0) at the centre of the top-left pixel; or null if the image or parameters
     *     are invalid
     */
    public static native double[] detect(
            long handle,
            long dataAddr,
            int width,
            int height,
            int stride,
            int minTagPx,
            int maxHamming,
            int minContrast,
            int minBorderCells,
            int sharpeningX100);
}
