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

  napa_params params{minTagPx, maxHamming, minContrast, minBorderCells,
                     sharpeningX100};
  std::vector<napa_detection> detections(kMaxDetections);
  int count =
      napa_detect(detector, reinterpret_cast<const uint8_t*>(dataAddr), width,
                  height, stride, &params, detections.data(), kMaxDetections);
  if (count < 0) {
    return nullptr;
  }

  std::vector<double> values;
  values.reserve(static_cast<size_t>(count) * kValuesPerDetection);
  for (int i = 0; i < count; i++) {
    const napa_detection& detection = detections[i];
    values.push_back(detection.id);
    values.push_back(detection.hamming);
    values.push_back(detection.decision_margin);
    values.insert(values.end(), detection.corners, detection.corners + 8);
  }

  jdoubleArray result = env->NewDoubleArray(values.size());
  if (!result) {
    return nullptr;
  }
  env->SetDoubleArrayRegion(result, 0, values.size(), values.data());
  return result;
}

}  // extern "C"
