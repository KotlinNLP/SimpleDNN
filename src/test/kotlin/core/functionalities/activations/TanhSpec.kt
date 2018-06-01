/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */

class TanhSpec : Spek({

  describe("a Tanh activation function") {

    val activationFunction = Tanh()
    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val activatedArray = activationFunction.f(array)

    on("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.0, 0.099668, 0.00999967, -0.099668, -0.00999967, 0.76159416, 1.0, -0.76159416, -1.0
      ))

      it("should return the expected errors") {
        assertEquals(true, expectedArray.equals(activatedArray, tolerance = 1.0e-08))
      }
    }

    on("dfOptimized") {

      val dfArray = activationFunction.dfOptimized(activatedArray)
      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          1.0, 0.99006629, 0.99990001, 0.99006629, 0.999900007, 0.41997434, 8.2e-09, 0.41997434, 8.2e-09
      ))

      it("should return the expected errors") {
        assertEquals(true, expectedArray.equals(dfArray, tolerance = 1.0e-08))
      }
    }
  }
})
