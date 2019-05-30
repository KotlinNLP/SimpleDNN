/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.HardSigmoid
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class HardSigmoidSpec : Spek({

  describe("an HardSigmoid activation function") {

    val activationFunction = HardSigmoid()
    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.5, 0.52, 0.502, 0.48, 0.498, 0.7, 1.0, 0.3, 0.0
      ))

      it("should return the expected errors") {
        assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-08) }
      }
    }

    context("dfOptimized") {

      val dfArray = activationFunction.dfOptimized(activatedArray)
      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0
      ))

      it("should return the expected errors") {
        assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-08) }
      }
    }
  }
})
