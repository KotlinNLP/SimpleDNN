/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.utils.FeedforwardLayerStructureUtils
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

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

      on("input size 4 and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        layer.forward()

        it("should match the expected output values") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.79688, 0.0, 0.70137, -0.18775)),
            tolerance = 1.0e-05))
        }
      }

      on("input size 5 (activated with tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.inputArray.setActivation(Tanh())
        layer.inputArray.activate()
        layer.forward()

        it("should match the expected output values") {

          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18504, 0.29346, 0.5215)),
            tolerance = 1.0e-05))
        }
      }

      on("input size 5 and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        layer.forward()

        it("should match the expected output values") {

          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18687, 0.28442, 0.52871)),
            tolerance = 1.0e-05))
        }
      }
    }

    context("forward with relevance") {

      on("dense input") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val contributes = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.forward(paramsContributes = contributes)

        it("should match the expected output values") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18687, 0.28442, 0.52871)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected contributes") {
          val wContr: DenseNDArray = contributes.weights.values as DenseNDArray
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(-0.42, 0.54, -0.1, -0.8, -0.08),
                doubleArrayOf(-0.34, -0.46, 0.02, 0.44, -0.1),
                doubleArrayOf(0.08, 0.04, 0.04, 0.04, -0.02)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 3))
        layer.calculateRelevance(paramsContributes = contributes)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance.values is DenseNDArray }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance.values as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.55888, 0.20978, 0.09943, 0.05652, 0.07539)),
              tolerance = 1.0e-05)
          }
        }
      }

      on("sparse input") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53SparseBinary()
        val contributes = FeedforwardLayerParameters(inputSize = 5, outputSize = 3, sparseInput = true)

        layer.forward(paramsContributes = contributes)

        it("should match the expected output values") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1769, 0.53144, 0.29166)),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 3))
        layer.calculateRelevance(paramsContributes = contributes)

        it("should set a Sparse input relevance") {
          assertTrue { layer.inputArray.relevance.values is SparseNDArray }
        }

        it("should match the expected input relevance") {
          val relevance: SparseNDArray = layer.inputArray.relevance.values as SparseNDArray
          assertTrue {
            relevance.equals(
              SparseNDArrayFactory.arrayOf(
                activeIndicesValues = arrayOf(
                  SparseEntry(Indices(2, 0), 1.04945),
                  SparseEntry(Indices(4, 0), -0.04945)
                ),
                shape = Shape(5)
              ),
              tolerance = 1.0e-05)
          }
        }
      }
    }

    context("backward") {

      on("input size 4 and output size 5 (tanh)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer45()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold5()
        val paramsErrors = FeedforwardLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -1.29688, 0.4, 1.60137, -1.08775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -1.29688, 0.4, 1.60137, -1.08775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, (paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.31754, 0.35724, 0.35724, -0.39693),
              doubleArrayOf(1.0375, 1.16719, 1.16719, -1.29688),
              doubleArrayOf(-0.32, -0.36, -0.36, 0.4),
              doubleArrayOf(-1.2811, -1.44124, -1.44124, 1.60137),
              doubleArrayOf(0.8702, 0.97897, 0.97897, -1.08775)
            )),
            tolerance = 0.010))
        }

        it("should match the expected errors of the inputArray") {

          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01972, -2.52839, 1.06928, 0.44533)),
            tolerance = 0.010))
        }
      }

      on("input size 5 (activated with tanh) and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()
        val paramsErrors = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.inputArray.setActivation(Tanh())
        layer.inputArray.activate()
        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81496, 0.29346, 0.5215)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81496, 0.29346, 0.5215)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, (paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.30964, 0.54116, 0.0, -0.49254, 0.16085),
              doubleArrayOf(-0.1115, -0.19487, 0.0, 0.17736, -0.05792),
              doubleArrayOf(-0.19814, -0.34629, 0.0, 0.31518, -0.10293)
            )),
            tolerance = 1.0e-05)) // some value is rounded exactly with the 3rd digit equal to 5
        }

        it("should match the expected errors of the inputArray") {

          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.37648, 0.46292, -0.37159, 0.62905, 0.39789)),
            tolerance = 1.0e-05))
        }
      }

      on("input size 5 and output size 3 (softmax)") {

        val layer = FeedforwardLayerStructureUtils.buildLayer53()
        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()
        val paramsErrors = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81313, 0.28442, 0.52871)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertEquals(true, paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81313, 0.28442, 0.52871)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertEquals(true, (paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.32525, 0.6505, 0.0, -0.56919, 0.16263),
              doubleArrayOf(-0.11377, -0.22753, 0.0, 0.19909, -0.05688),
              doubleArrayOf(-0.21148, -0.42297, 0.0, 0.37010, -0.10574)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {

          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4474, 0.82115, -0.37411, 0.98377, 0.41057)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
