/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.functionalities.gradientclipping.GradientClipping
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class GradientClippingSpec: Spek({

  describe("the gradient clipping") {

    context("clip at value") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()
      val gradientClipping = GradientClipping()
      gradientClipping.clipByValue(paramsErrors, 0.7)

      it("should match the expected parameters at index 0") {
        assertTrue {
          paramsErrors[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  doubleArrayOf(0.5, 0.6, -0.7, -0.6),
                  doubleArrayOf(0.7, -0.4, 0.1, -0.7),
                  doubleArrayOf(0.7, -0.7, 0.3, 0.5),
                  doubleArrayOf(0.7, -0.7, 0.0, -0.1),
                  doubleArrayOf(0.4, 0.7, -0.7, 0.7)
              )),
              tolerance = 1.0e-6)
        }
      }

      it("should match the expected parameters at index 1") {
        assertTrue {
          paramsErrors[1].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.7, 0.4, 0.7, 0.1)),
              tolerance = 1.0e-6)
        }
      }
    }

    context("clip at 2-norm") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()
      val gradientClipping = GradientClipping()
      gradientClipping.clipByNorm(paramsErrors, 2.0, "2")

      it("should match the expected parameters at index 0") {
        assertTrue {
          paramsErrors[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  doubleArrayOf(0.314814, 0.377777, -0.503702, -0.377777),
                  doubleArrayOf(0.440739, -0.251851, 0.062962, -0.503702),
                  doubleArrayOf(0.440739, -0.440739, 0.188888, 0.314814),
                  doubleArrayOf(0.503702, -0.566665, 0.0, -0.062962),
                  doubleArrayOf(0.251851, 0.629628,  -0.440739, 0.503702)
              )),
              tolerance = 1.0e-6)
        }
      }

      it("should match the expected parameters at index 1") {
        assertTrue {
          paramsErrors[1].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.566665, 0.440739, 0.251851, 0.503702, 0.062962)),
              tolerance = 1.0e-6)
        }
      }
    }

    context("clip at inf-norm") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()
      val gradientClipping = GradientClipping()
      gradientClipping.clipByNorm(paramsErrors, 0.5, "inf")

      it("should match the expected parameters at index 0") {
        assertTrue {
          paramsErrors[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  doubleArrayOf(0.25, 0.3, -0.4, -0.3),
                  doubleArrayOf(0.35, -0.2, 0.05, -0.4),
                  doubleArrayOf(0.35, -0.35, 0.15, 0.25),
                  doubleArrayOf(0.4, -0.45, 0.0,	-0.05),
                  doubleArrayOf(0.2, 0.5, -0.35, 0.4)
              )),
              tolerance = 1.0e-6)
        }
      }

      it("should match the expected parameters at index 1") {
        assertTrue {
          paramsErrors[1].values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.45, 0.35, 0.2, 0.4, 0.05)),
              tolerance = 1.0e-6)
        }
      }
    }
  }
})
