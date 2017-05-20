/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package activations

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class SoftmaxSpec : Spek({

  describe("a Softmax activation function") {

    val activationFunction = Softmax()
    val array = NDArray.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = NDArray.arrayOf(doubleArrayOf(
        4.53832e-05, 5.01562e-05, 4.58394e-05, 4.10645e-05,
        4.49317e-05, 1.233645e-04, 9.996325629e-01, 1.66956e-05, 2.1e-09
      ))

      it("should have 1.0 as the sum its element") {
        assertEquals(1.0, activatedArray.sum())
      }

      it("should return the expected array") {
        assertEquals(true, expectedArray.equals(activatedArray, tolerance = 1.0e-10))
      }
    }

    context("df") {

      on("returning a new NDArray as output") {

        val dfArray = activationFunction.dfOptimized(activatedArray)

        it("should return the expected array") {
          assertEquals(true, activatedArray.equals(dfArray, tolerance = 1.0e-08))
        }
      }

      on("assigning the results to an output array") {

        val outDfArray = NDArray.emptyArray(array.shape)
        activationFunction.dfOptimized(activatedArray, outDfArray)

        it("should assign the expected values") {
          assertEquals(true, activatedArray.equals(outDfArray, tolerance = 1.0e-08))
        }
      }
    }
  }
})
