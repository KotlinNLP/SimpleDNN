/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.losses

import com.kotlinnlp.simplednn.core.functionalities.losses.AugmentedMSECalculator
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
class AugmentedMSECalculatorSpec : Spek({

  describe("an AugmentedMSECalculator") {

    val outputValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3))
    val goldValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.2, 0.1, 0.0))

    context("with loss partition disabled") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.0)

      on("calculateLoss") {

        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.045, 0.005, 0.005, 0.045)).equals(loss))
        }
      }

      on("calculateErrors") {

        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.1, 0.1, 0.3)).equals(errors))
        }
      }
    }

    context("with none injected errors") {

      val lossCalculator = AugmentedMSECalculator()

      on("calculateLoss") {

        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0405, 0.0045, 0.0045, 0.0405)).equals(loss))
        }
      }

      on("calculateErrors") {

        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.27, -0.09, 0.09, 0.27)).equals(errors))
        }
      }
    }

    context("with hard injected errors") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.9, c = 15.0)
      lossCalculator.injectedErrorStrength = AugmentedMSECalculator.InjectedErrorStrength.HARD

      on("calculateLoss") {

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0045, 0.005, 0.01849999, 0.04499998))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08))
        }
      }

      on("calculateErrors") {

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03, 0.07999997, 0.18999995, 0.29999992))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08))
        }
      }
    }

    context("with medium injected errors") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.9, c = 15.0)
      lossCalculator.injectedErrorStrength = AugmentedMSECalculator.InjectedErrorStrength.MEDIUM

      on("calculateLoss") {

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0045, 0.00321587, 0.01136348, 0.02894283))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08))
        }
      }

      on("calculateErrors") {

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03, 0.05991829, 0.14983657, 0.23975486))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08))
        }
      }
    }

    context("with low injected errors") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.9, c = 15.0)
      lossCalculator.injectedErrorStrength = AugmentedMSECalculator.InjectedErrorStrength.SOFT

      on("calculateLoss") {

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0045, 0.00058731, 0.00084924, 0.00528579))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08))
        }
      }

      on("calculateErrors") {

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03, 0.00253628, 0.03507256, 0.06760885))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08))
        }
      }
    }
  }
})
