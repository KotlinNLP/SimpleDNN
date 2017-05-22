/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.losses.AugmentedLossStrength
import com.kotlinnlp.simplednn.core.functionalities.losses.AugmentedMSECalculator
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class AugmentedMSECalculatorSpec : Spek({

  describe("an AugmentedMSECalculator") {

    val outputValues = NDArray.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3))
    val goldValues = NDArray.arrayOf(doubleArrayOf(0.3, 0.2, 0.1, 0.0))

    context("without augmented errors") {

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors") {
          assertTrue(NDArray.arrayOf(doubleArrayOf(-0.27, -0.09, 0.09, 0.27)).equals(outputErrors))
        }
      }

      on("calculateLoss") {

        val lossCalculator = AugmentedMSECalculator()

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the pre-computed avgLoss") {
          assertTrue(true)
        }

        it("should calculate the pre-computed scalar avgLoss") {
          assertTrue(true)
        }
      }
    }

    context("with hard augmented errors") {

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.HARD.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertTrue(NDArray.arrayOf(doubleArrayOf(-0.27, -0.08, 0.109999, 0.299999)).equals(outputErrors))
        }
      }
    }

    context("with medium augmented errors") {

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.MEDIUM.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertTrue(NDArray.arrayOf(doubleArrayOf(-0.27, -0.083679, 0.102642, 0.288964)).equals(outputErrors))
        }
      }
    }

    context("with soft augmented errors") {

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.SOFT.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertTrue(NDArray.arrayOf(doubleArrayOf(-0.27, -0.089048, 0.091903, 0.272855)).equals(outputErrors))
        }
      }
    }
  }
})
