/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.losses

import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class SoftmaxCrossEntropyCalculatorSpec : Spek({

  describe("a SoftmaxCrossEntropyCalculator") {

    val lossCalculator = SoftmaxCrossEntropyCalculator

    context("output and gold not compatible") {

      val outputValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3))
      val goldValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.2, 0.1, 0.0))

      it("should raise an error when calling the calculateLoss() method") {
        assertFails { lossCalculator.calculateLoss(outputValues, goldValues) }
      }
    }

    context("single output") {

      val outputValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.7))
      val goldValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 1.0, 0.0))

      context("calculateErrors") {
        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, -0.8, 0.7)).equals(errors, tolerance = 1.0e-06)
          }
        }
      }

      context("calculateLoss") {
        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 1.609438, 0.0)).equals(loss, tolerance = 1.0e-06)
          }
        }
      }
    }

    context("output sequence") {

      val outputValuesSequence: List<DenseNDArray> = listOf(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.1, 0.4, 0.7))
      )
      val goldValuesSequence: List<DenseNDArray> = listOf(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 1.0, 0.0)),
        DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.0, 0.0, 0.0))
      )

      context("calculateErrors of a sequence of length 2") {

        val sequenceErrors: List<DenseNDArray> = lossCalculator.calculateErrors(outputValuesSequence, goldValuesSequence)

        it("should return an array of length 2") {
          assertEquals(2, sequenceErrors.size)
        }

        it("should calculate the expected errors of the first element") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, -0.8, 0.3))
              .equals(sequenceErrors[0], tolerance = 1.0e-06)
          }
        }

        it("should calculate the expected errors of the second element") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1, 0.4, 0.7))
              .equals(sequenceErrors[1], tolerance = 1.0e-06)
          }
        }
      }

      context("calculateMeanLoss of a sequence of length 2") {
        val meanLoss = lossCalculator.calculateMeanLoss(outputValuesSequence, goldValuesSequence)

        it("should calculate the expected mean loss") {
          assertTrue { equals(0.28782314, meanLoss, tolerance = 1.0e-08) }
        }
      }
    }
  }
})
