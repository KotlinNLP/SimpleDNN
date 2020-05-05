/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.GeLU
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class GeLUSpec: Spek({

  describe("a GeLU activation function") {

    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))

    context("default configuration") {

      val activationFunction = GeLU
      val activatedArray = activationFunction.f(array)

      context("f") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.0, 0.053983, 0.00504, -0.046017, -0.00496, 0.841192, 10.0, -0.158808, 0.0
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-06) }
        }
      }

      context("df") {

        val dfArray = activationFunction.df(array)
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.5, 0.579522, 0.507979, 0.420478, 0.492021, 1.082964, 1.0, -0.082964, 0.0
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-06) }
        }
      }
    }
  }
})
