/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.utils.SimpleRecurrentLayerStructureUtils
import layers.structure.contextwindows.SimpleRecurrentLayerContextWindow
import kotlin.test.assertEquals

/**
 *
 */
class SimpleRecurrentLayerStructureSpec : Spek({

  describe("a SimpleRecurrentLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.4, -0.8, 0.0, 0.7, -0.19)),
            tolerance = 0.005))
        }
      }

      on("with previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Back())
        layer.forward()

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.74, -0.80, 0.20, 0.91, 0.14)),
            tolerance = 0.005))
        }
      }
    }

    context("backward") {

      on("without next state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Empty())
        val paramsErrors = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.97, -1.55, 0.15, -0.94, -0.64)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.97, -1.55, 0.15, -0.94, -0.64)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, paramsErrors.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.77, 0.87, 0.87, -0.97),
              doubleArrayOf(1.24, 1.39, 1.39, -1.55),
              doubleArrayOf(-0.12, -0.14, -0.14, 0.15),
              doubleArrayOf(0.75, 0.84, 0.84, -0.94),
              doubleArrayOf(0.51, 0.57, 0.57, -0.64)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the recurrent weights") {
          assertEquals(true, paramsErrors.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-2.47, 0.14, 1.11, 1.48)),
            tolerance = 0.005))
        }
      }

      on("with next state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Front())
        val paramsErrors = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-0.75, -1.69, 0.41, -0.98, -1.48)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.75, -1.69, 0.41, -0.98, -1.48)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, paramsErrors.weights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(0.6, 0.67, 0.67, -0.75),
              doubleArrayOf(1.35, 1.52, 1.52, -1.69),
              doubleArrayOf(-0.33, -0.37, -0.37, 0.41),
              doubleArrayOf(0.78, 0.88, 0.88, -0.98),
              doubleArrayOf(1.18, 1.33, 1.33, -1.48)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the recurrent weights") {
          assertEquals(true, paramsErrors.recurrentWeights.values.equals(
            NDArray.arrayOf(arrayOf(
              doubleArrayOf(-0.04, -0.08, 0.0, 0.07, -0.02),
              doubleArrayOf(-0.04, -0.08, 0.0, 0.07, -0.02),
              doubleArrayOf(0.2, 0.4, 0.0, -0.35, 0.09),
              doubleArrayOf(-0.28, -0.56, 0.0, 0.49, -0.13),
              doubleArrayOf(-0.08, -0.16, 0.0, 0.14, -0.04)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            NDArray.arrayOf(doubleArrayOf(-2.65, -0.65, 1.59, 0.92)),
            tolerance = 0.005))
        }
        }
    }
  }
})
