/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.Softplus
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class SoftplusSpec : Spek({

  describe("a Softplus activation function") {

    val activationFunction = Softplus()
    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.693147181, 0.74439666, 0.69815968, 0.64439666, 0.688159681, 1.313261688, 10.0000454, 0.313261687, 4.539e-5
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-08) }
      }
    }

    context("dfOptimized") {

      val dfArray = activationFunction.dfOptimized(activatedArray)
      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.5, 0.52497919, 0.50249998, 0.47502081, 0.49750002, 0.73105858, 0.9999546, 0.26894142, 4.54e-5
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-08) }
      }
    }
  }
})
