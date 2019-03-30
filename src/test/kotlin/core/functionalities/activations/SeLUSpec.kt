/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.SeLU
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

class SeLUSpec: Spek({

  describe("a SeLU activation function") {

    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))

    context("default configuration") {

      val activationFunction = SeLU()
      val activatedArray = activationFunction.f(array)

      on("f") {

        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
            0.0, 0.10507009, 0.01050700, -0.16730527, -0.01749338, 1.0507009, 10.5070099, -1.11133072, -1.758019508
        ))

        it("should return the expected errors") {
          assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-07) }
        }
      }

      on("dfOptimized") {

        val dfArray = activationFunction.dfOptimized(activatedArray)
        val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
            1.758099343, 1.0507009, 1.0507009, 1.59079407, 1.740605962, 1.0507009, 1.0507009, 0.646768603, 0.000079817586
        ))

        it("should return the expected errors") {
          assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-07) }
        }
      }
    }
  }
})
