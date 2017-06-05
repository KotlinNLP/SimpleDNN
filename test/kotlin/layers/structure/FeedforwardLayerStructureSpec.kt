/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.utils.FeedforwardLayerStructureUtils
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 *
 */
class FeedforwardLayerStructureSpec : Spek({

  describe("a FeedForwardLayerStructure") {

    context("initialization") {

      on("before calling any method") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()

        it("should contain null paramsErrors") {
          assertNull(layer.paramsErrors)
        }
      }
    }

    context("forward") {

      on("input size 4 (no activation) and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        layer.forward()

        it("should match the expected output values") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.8, 0.0, 0.7, -0.19)),
            tolerance = 0.005))
        }
      }

      on("input size 5 (tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.forward()

        it("should match the expected output values") {

          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.19, 0.29, 0.53)),
            tolerance = 0.005))
        }
      }

      on("input size 5 and output size 3 (without activation functions)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53NoActivation()
        layer.forward()

        it("should match the expected output values") {

          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.82, -0.52, 0.19)),
            tolerance = 0.005))
        }
      }
    }

    context("backward") {

      val lossCalculator = MSECalculator()

      on("input size 4 (no activation) and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold5()
        val paramsErrors = FeedforwardLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = lossCalculator.calculateErrors(output = layer.outputArray.values, outputGold = outputGold)
        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.38, 0.3, -0.37, 0.5, 0.4)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.38, 0.3, -0.37, 0.5, 0.4)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, (paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.3, 0.34, 0.34, -0.38),
              doubleArrayOf(-0.24, -0.27, -0.27, 0.3),
              doubleArrayOf(0.3, 0.34, 0.34, -0.37),
              doubleArrayOf(-0.4, -0.45, -0.45, 0.5),
              doubleArrayOf(-0.32, -0.36, -0.36, 0.4)
            )),
            tolerance = 0.010))
        }

        it("should match the expected errors of the inputArray") {

          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.32, -0.14, -0.06, 0.07)),
            tolerance = 0.010))
        }
      }

      on("input size 5 (tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()
        val paramsErrors = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.forward()

        val errors = lossCalculator.calculateErrors(output = layer.outputArray.values, outputGold = outputGold)
        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81, 0.29, 0.53)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81, 0.29, 0.53)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, (paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.32, 0.65, 0.0, -0.57, 0.15),
              doubleArrayOf(-0.11, -0.23, 0.0, 0.2, -0.05),
              doubleArrayOf(-0.21, -0.42, 0.0, 0.37, -0.1)
            )),
            tolerance = 0.006)) // some value is rounded exactly with the 3rd digit equal to 5
        }

        it("should match the expected errors of the inputArray") {

          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.38, 0.3, -0.37, 0.5, 0.4)),
            tolerance = 0.005))
        }
      }

      on("input size 5 and output size 3 (no activation functions)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53NoActivation()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()
        val paramsErrors = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.forward()

        val errors = lossCalculator.calculateErrors(output = layer.outputArray.values, outputGold = outputGold)
        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.82, -0.52, 0.19)),
            tolerance = 0.006))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.82, -0.52, 0.19)),
            tolerance = 0.006))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, (paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.76, 1.98, 0.0, -1.58, 0.34),
              doubleArrayOf(0.22, 0.57, 0.0, -0.46, 0.1),
              doubleArrayOf(-0.08, -0.2, 0.0, 0.16, -0.04)
            )),
            tolerance = 0.006))
        }

        it("should match the expected errors of the inputArray") {

          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.94, 1.14, -1.94, 1.5, -0.08)),
            tolerance = 0.005))
        }
      }
    }
  }
})
