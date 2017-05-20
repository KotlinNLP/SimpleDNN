/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.losses.AugmentedLossStrength
import com.kotlinnlp.simplednn.core.functionalities.losses.AugmentedMSECalculator
import com.kotlinnlp.simplednn.core.functionalities.losses.MulticlassMSECalculator
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
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
class LossCalculatorSpec : Spek({

  describe("a LossCalculator") {

    /**
     * MSECalculator
     */
    context("MSECalculator") {

      val lossCalculator = MSECalculator()

      val outputValues = NDArray.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3))
      val goldValues = NDArray.arrayOf(doubleArrayOf(0.3, 0.2, 0.1, 0.0))

      on("calculateErrors") {
        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(-0.3, -0.1, 0.1, 0.3)).equals(outputErrors))
        }
      }


      on("calculateLoss") {
        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the pre-computed avgLoss"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(0.045, 0.005, 0.005, 0.045)).equals(outputLoss))
        }

        it("should calculate the pre-computed scalar avgLoss") {
          assertEquals(0.025, outputLoss.avg())
        }

      }
    }

    /**
     * MulticlassMSECalculator
     */
    context("MulticlassMSECalculator") {

      val lossCalculator = MulticlassMSECalculator()

      val outputValues = NDArray.arrayOf(doubleArrayOf(0.0, 0.7, 0.2, 0.1))
      val goldValues = NDArray.arrayOf(doubleArrayOf(1.0, 0.0, 0.0, 0.0))

      on("calculateErrors") {
        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(-1.0, 0.7, 0.2, 0.1)).equals(outputErrors))
        }
      }

      on("calculateLoss") {
        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)

        //println(outputLoss)
        // TODO
        it("should calculate the pre-computed avgLoss"){
          assertEquals(true, true)
        }

        // TODO
        it("should calculate the pre-computed scalar avgLoss") {
          assertEquals(true, true)
        }

      }
    }

    /**
     * MSECalculator
     */
    context("AugmentedMSECalculator") {

      val outputValues = NDArray.arrayOf(doubleArrayOf(0.0, 0.1, 0.2, 0.3))
      val goldValues = NDArray.arrayOf(doubleArrayOf(0.3, 0.2, 0.1, 0.0))

      on("calculateErrors without augmented errors") {

        val lossCalculator = AugmentedMSECalculator()

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(-0.27, -0.09, 0.09, 0.27)).equals(outputErrors))
        }
      }

      on("calculateLoss without augmented errors") {

        val lossCalculator = AugmentedMSECalculator()

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the pre-computed avgLoss"){
          assertEquals(true, true)
        }

        it("should calculate the pre-computed scalar avgLoss") {
          assertEquals(true, true)
        }

      }

      on("calculateErrors with hard augmented errors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.augmentedError = AugmentedLossStrength.HARD.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(-0.27, -0.08, 0.109999, 0.299999)).equals(outputErrors))
        }
      }

      on("calculateErrors with medium augmented errors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.augmentedError = AugmentedLossStrength.MEDIUM.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(-0.27, -0.083679, 0.102642, 0.288964)).equals(outputErrors))
        }
      }

      on("calculateErrors with soft augmented errors") {

        val lossCalculator = AugmentedMSECalculator()

        lossCalculator.augmentedError = AugmentedLossStrength.SOFT.weight

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the pre-computed output errors"){
          assertEquals(true, NDArray.arrayOf(doubleArrayOf(-0.27, -0.089048, 0.091903, 0.272855)).equals(outputErrors))
        }
      }

    }

  }

})
