/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class MSECalculatorSpec : Spek({

  describe("a MSECalculator") {

    val lossCalculator = MSECalculator()

    val outputValues = NDArray.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3))
    val goldValues = NDArray.arrayOf(doubleArrayOf(0.3, 0.2, 0.1, 0.0))

    on("calculateErrors") {
      val errors = lossCalculator.calculateErrors(outputValues, goldValues)

      it("should calculate the expected errors"){
        assertTrue(NDArray.arrayOf(doubleArrayOf(-0.3, -0.1, 0.1, 0.3)).equals(errors))
      }
    }

    on("calculateLoss") {
      val loss = lossCalculator.calculateLoss(outputValues, goldValues)

      it("should calculate the expected loss"){
        assertTrue(NDArray.arrayOf(doubleArrayOf(0.045, 0.005, 0.005, 0.045)).equals(loss))
      }
    }
  }
})
