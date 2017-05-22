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

      on("calculateLoss") {

        val lossCalculator = AugmentedMSECalculator()

        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue(NDArray.arrayOf(doubleArrayOf(0.0405, 0.0045, 0.0045, 0.0405)).equals(loss))
        }
      }

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue(NDArray.arrayOf(doubleArrayOf(-0.27, -0.09, 0.09, 0.27)).equals(errors))
        }
      }
    }

    context("with hard augmented errors") {

      on("calculateLoss") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.HARD.weight

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = NDArray.arrayOf(doubleArrayOf(0.0405, 0.00499995, 0.00649982, 0.04499959))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08))
        }
      }

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.HARD.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = NDArray.arrayOf(doubleArrayOf(-0.27, -0.08000045, 0.10999909, 0.29999864))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08))
        }
      }
    }

    context("with medium augmented errors") {

      on("calculateLoss") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.MEDIUM.weight

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = NDArray.arrayOf(doubleArrayOf(0.0405, 0.00469979, 0.00529915, 0.04229809))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08))
        }
      }

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.MEDIUM.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = NDArray.arrayOf(doubleArrayOf(-0.27, -0.08367879, 0.10264241, 0.28896362))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08))
        }
      }
    }

    context("with low augmented errors") {

      on("calculateLoss") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.SOFT.weight

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = NDArray.arrayOf(doubleArrayOf(0.0405, 0.00450453, 0.00451811, 0.04054075))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08))
        }
      }

      on("calculateErrors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.injectedError = AugmentedLossStrength.SOFT.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = NDArray.arrayOf(doubleArrayOf(-0.27, -0.08904837, 0.09190325, 0.27285488))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08))
        }
      }
    }
  }
})
