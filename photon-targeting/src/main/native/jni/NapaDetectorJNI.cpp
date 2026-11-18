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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../napa/napa_core.h"
#include "org_photonvision_jni_NapaDetectorJni.h"

namespace {
// Tags beyond this many in one frame are dropped. PhotonVision reports at most
// 127 targets, but its stress-test image alone has 280.
constexpr int kMaxDetections = 1024;
constexpr int kValuesPerDetection =
    org_photonvision_jni_NapaDetectorJni_kValuesPerDetection;
static_assert(org_photonvision_jni_NapaDetectorJni_kLensNone == NAPA_LENS_NONE);
static_assert(org_photonvision_jni_NapaDetectorJni_kLensBrownConrady ==
              NAPA_LENS_BROWN_CONRADY);
}  // namespace

extern "C" {

/*
 * Class:     org_photonvision_jni_NapaDetectorJni
 * Method:    create
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_jni_NapaDetectorJni_create
  (JNIEnv* env, jclass, jstring family)
{
  const char* name = env->GetStringUTFChars(family, nullptr);
  if (!name) {
    return 0;
  }
  napa_detector* detector = napa_create_family(name);
  env->ReleaseStringUTFChars(family, name);
  return reinterpret_cast<jlong>(detector);
}

/*
 * Class:     org_photonvision_jni_NapaDetectorJni
 * Method:    recommendedHamming
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL
Java_org_photonvision_jni_NapaDetectorJni_recommendedHamming
  (JNIEnv* env, jclass, jstring family)
{
  const char* name = env->GetStringUTFChars(family, nullptr);
  if (!name) {
    return -1;
  }
  napa_family_layout info;
  int status = napa_family_info(name, &info);
  env->ReleaseStringUTFChars(family, name);
  return status == 0 ? info.recommended_hamming : -1;
}

/*
 * Class:     org_photonvision_jni_NapaDetectorJni
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_jni_NapaDetectorJni_destroy
  (JNIEnv*, jclass, jlong handle)
{
  napa_destroy(reinterpret_cast<napa_detector*>(handle));
}

/*
 * Class:     org_photonvision_jni_NapaDetectorJni
 * Method:    setThreads
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL
Java_org_photonvision_jni_NapaDetectorJni_setThreads
  (JNIEnv*, jclass, jlong handle, jint threads)
{
  auto* detector = reinterpret_cast<napa_detector*>(handle);
  return detector ? napa_set_threads(detector, threads) : -1;
}

/*
 * Class:     org_photonvision_jni_NapaDetectorJni
 * Method:    setLens
 * Signature: (JI[D[D)I
 */
JNIEXPORT jint JNICALL
Java_org_photonvision_jni_NapaDetectorJni_setLens
  (JNIEnv* env, jclass, jlong handle, jint model, jdoubleArray intrinsics,
   jdoubleArray coefficients)
{
  auto* detector = reinterpret_cast<napa_detector*>(handle);
  if (!detector) {
    return -1;
  }
  if (model == NAPA_LENS_NONE) {
    return napa_set_lens(detector, NAPA_LENS_NONE, nullptr, nullptr, 0);
  }
  if (!intrinsics || !coefficients || env->GetArrayLength(intrinsics) != 5) {
    return -1;
  }
  double k[5];
  env->GetDoubleArrayRegion(intrinsics, 0, 5, k);
  std::vector<double> d(env->GetArrayLength(coefficients));
  env->GetDoubleArrayRegion(coefficients, 0, d.size(), d.data());
  return napa_set_lens(detector, model, k, d.data(),
                       static_cast<int>(d.size()));
}

/*
 * Class:     org_photonvision_jni_NapaDetectorJni
 * Method:    detect
 * Signature: (JJIIIIIIII)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_org_photonvision_jni_NapaDetectorJni_detect
  (JNIEnv* env, jclass, jlong handle, jlong dataAddr, jint width, jint height,
   jint stride, jint minTagPx, jint maxHamming, jint minContrast,
   jint minBorderCells, jint sharpeningX100)
{
  auto* detector = reinterpret_cast<napa_detector*>(handle);
  if (!detector || dataAddr == 0) {
    return nullptr;
  }

  const napa_params params{.min_tag_px = minTagPx,
                           .max_hamming = maxHamming,
                           .min_contrast = minContrast,
                           .min_border_cells = minBorderCells,
                           .sharpening_x100 = sharpeningX100,
                           // 2 is NAPA's "auto", its Python default: re-detect
                           // only frames where a decoded tag touches dark
                           // shapes at its corners.
                           .split_touching = 2};
  // One buffer per calling thread, reused across frames: the results are
  // copied out before the call returns.
  thread_local std::vector<napa_detection> detections(kMaxDetections);
  int count =
      napa_detect(detector, reinterpret_cast<const uint8_t*>(dataAddr), width,
                  height, stride, &params, detections.data(), kMaxDetections);
  if (count < 0) {
    return nullptr;
  }

  jdoubleArray result = env->NewDoubleArray(count * kValuesPerDetection);
  if (!result) {
    return nullptr;
  }
  for (int i = 0; i < count; i++) {
    const napa_detection& detection = detections[i];
    double values[kValuesPerDetection] = {
        static_cast<double>(detection.id),
        static_cast<double>(detection.hamming), detection.decision_margin};
    std::copy(detection.corners, detection.corners + 8, values + 3);
    env->SetDoubleArrayRegion(result, i * kValuesPerDetection,
                              kValuesPerDetection, values);
  }
  return result;
}

}  // extern "C"
