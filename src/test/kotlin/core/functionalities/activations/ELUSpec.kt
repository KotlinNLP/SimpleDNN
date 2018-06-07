/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */

class ELUSpec : Spek({

  describe("an ELU activation function") {

    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))

    context("alpha = 1.0") {

      val activationFunction = ELU(alpha = 1.0)
      val activatedArray = activationFunction.f(array)

      on("f") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.0, 0.1, 0.01, -0.095162582, -0.009950166, 1.0, 10.0, -0.632120559, -0.9999546
        ))

        it("should return the expected errors") {
          assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-08) }
        }
      }

      on("dfOptimized") {

        val dfArray = activationFunction.dfOptimized(activatedArray)
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          1.0, 1.0, 1.0, 0.904837418, 0.990049834, 1.0, 1.0, 0.367879441, 0.0000454
        ))

        it("should return the expected errors") {
          assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-08) }
        }
      }
    }
  }
})
