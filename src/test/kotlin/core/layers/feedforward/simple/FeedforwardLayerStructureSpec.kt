/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.simple

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
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
class FeedforwardLayerStructureSpec : Spek({

  describe("a FeedForwardLayerStructure") {

    context("forward") {

      on("input size 4 and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        layer.forward()

        it("should match the expected output values") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.79688, 0.0, 0.70137, -0.18775)),
              tolerance = 1.0e-05)
          }
        }
      }

      on("input size 5 (activated with tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.inputArray.setActivation(Tanh())
        layer.inputArray.activate()
        layer.forward()

        it("should match the expected output values") {

          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18504, 0.29346, 0.5215)),
              tolerance = 1.0e-05)
          }
        }
      }

      on("input size 5 and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.forward()

        it("should match the expected output values") {

          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18687, 0.28442, 0.52871)),
              tolerance = 1.0e-05)
          }
        }
      }
    }

    context("forward with relevance") {

      on("dense input") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val contributions = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.forward(layerContributions = contributions)

        it("should match the expected output values") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18687, 0.28442, 0.52871)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.unit.weights.values
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.42, 0.54, -0.1, -0.8, -0.08),
                doubleArrayOf(-0.34, -0.46, 0.02, 0.44, -0.1),
                doubleArrayOf(0.08, 0.04, 0.04, 0.04, -0.02)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 3))
        layer.setInputRelevance(layerContributions = contributions)

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.55888, 0.20978, 0.09943, 0.05652, 0.07539)),
              tolerance = 1.0e-05)
          }
        }
      }

      // TODO: reintroduce tests for sparse input
    }

    context("backward") {

      on("input size 4 and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold5()

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as FeedforwardLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.33439, -0.47334, 0.4, 0.81362, -1.0494)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.33439, -0.47334, 0.4, 0.81362, -1.0494)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.unit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.26751, 0.30095, 0.30095, -0.33439),
                doubleArrayOf(0.37867, 0.42601, 0.42601, -0.47334),
                doubleArrayOf(-0.32, -0.36, -0.36, 0.4),
                doubleArrayOf(-0.65089, -0.73226, -0.73226, 0.81362),
                doubleArrayOf(0.83952, 0.94446, 0.94446, -1.04940)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the inputArray") {

          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0126, -2.07296, 1.07476, -0.14158)),
              tolerance = 1.0e-05)
          }
        }
      }

      on("input size 5 (activated with tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

        layer.inputArray.setActivation(Tanh())
        layer.inputArray.activate()
        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as FeedforwardLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81496, 0.29346, 0.5215)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81496, 0.29346, 0.5215)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.30964, 0.54116, 0.0, -0.49254, 0.16085),
                doubleArrayOf(-0.1115, -0.19487, 0.0, 0.17736, -0.05792),
                doubleArrayOf(-0.19814, -0.34629, 0.0, 0.31518, -0.10293)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the inputArray") {

          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.440004, 0.828046, -0.371592, 0.991038, 0.414023)),
              tolerance = 1.0e-05)
          }
        }
      }

      on("input size 5 and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as FeedforwardLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81313, 0.28442, 0.52871)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the biases") {
          assertTrue {
            paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81313, 0.28442, 0.52871)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.unit.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.32525, 0.6505, 0.0, -0.56919, 0.16263),
                doubleArrayOf(-0.11377, -0.22753, 0.0, 0.19909, -0.05688),
                doubleArrayOf(-0.21148, -0.42297, 0.0, 0.37010, -0.10574)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected errors of the inputArray") {

          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4474, 0.82115, -0.37411, 0.98377, 0.41057)),
              tolerance = 1.0e-05)
          }
        }
      }
    }
  }
})
