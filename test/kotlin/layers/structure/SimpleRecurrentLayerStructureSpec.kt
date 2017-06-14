/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import layers.structure.utils.SimpleRecurrentLayerStructureUtils
import layers.structure.contextwindows.SimpleRecurrentLayerContextWindow
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.8, 0.0, 0.7, -0.19)),
            tolerance = 0.005))
        }
      }

      on("with previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Back())
        layer.forward()

        it("should match the expected outputArray") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.74, -0.80, 0.20, 0.91, 0.14)),
            tolerance = 0.005))
        }
      }
    }

    context("forward with relevance") {

      on("without previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Empty())
        val contributes = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward(paramsContributes = contributes)

        it("should match the expected outputArray") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.79688, 0.0, 0.70137, -0.18775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributes") {
          val wContr: DenseNDArray = contributes.weights.values as DenseNDArray
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(-0.3, -0.44, 0.82, -0.5),
                doubleArrayOf(-0.56, 0.36, -0.09, -0.8),
                doubleArrayOf(-0.635, 0.555, -0.345, 0.425),
                doubleArrayOf(-0.44, 1.01, 0.2, 0.1),
                doubleArrayOf(-0.42, -1.0, 0.53, 0.7)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.calculateInputRelevance(paramsContributes = contributes)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-12.08396, 12.52343, -7.69489, 8.25543)),
              tolerance = 1.0e-05)
          }
        }

        it("should throw an Exception when calculating the recurrent relevance") {
          assertFailsWith <KotlinNullPointerException> {
            layer.calculateRecurrentRelevance(paramsContributes = contributes)
          }
        }
      }

      on("with previous state context") {

        val prevStateLayer = SimpleRecurrentLayerContextWindow.Back().getPrevStateLayer()
        val contextWindow = mock<LayerContextWindow>()
        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(contextWindow)
        val contributes = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevStateLayer()).thenReturn(prevStateLayer)

        layer.forward(paramsContributes = contributes)

        it("should match the expected outputArray") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.74428, -0.8005, 0.19738, 0.9087, 0.13909)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributes") {
          val wContr: DenseNDArray = contributes.weights.values as DenseNDArray
          assertTrue {
            wContr.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(-0.35, -0.49, 0.77, -0.55),
                doubleArrayOf(-0.56, 0.36, -0.09, -0.8),
                doubleArrayOf(-0.5975, 0.5925, -0.3075, 0.4625),
                doubleArrayOf(-0.54, 0.91, 0.1, 0.0),
                doubleArrayOf(-0.37, -0.95, 0.58, 0.75)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected recurrent contributes") {
          val wRecContr: DenseNDArray = contributes.recurrentWeights.values
          assertTrue {
            wRecContr.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.04, 0.2, -0.2, 0.94, 0.6),
                doubleArrayOf(0.14, -0.16, -0.06, 0.63, -0.56),
                doubleArrayOf(0.15, 0.15, -0.24, 0.42, -0.43),
                doubleArrayOf(0.08, 0.06, -0.07, 0.26, 0.72),
                doubleArrayOf(0.08, 0.08, -0.28, 0.05, 0.2)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.calculateInputRelevance(paramsContributes = contributes)
        layer.calculateRecurrentRelevance(paramsContributes = contributes)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.95605, -0.44375, 0.48543, 1.21457)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.23878, 0.3096, -0.60637, 0.57811, 0.17968)),
              tolerance = 1.0e-05)
          }
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
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.97, -1.55, 0.15, -0.94, -0.64)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.97, -1.55, 0.15, -0.94, -0.64)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.77, 0.87, 0.87, -0.97),
              doubleArrayOf(1.24, 1.39, 1.39, -1.55),
              doubleArrayOf(-0.12, -0.14, -0.14, 0.15),
              doubleArrayOf(0.75, 0.84, 0.84, -0.94),
              doubleArrayOf(0.51, 0.57, 0.57, -0.64)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-2.47, 0.14, 1.11, 1.48)),
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
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.75, -1.69, 0.41, -0.98, -1.48)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.75, -1.69, 0.41, -0.98, -1.48)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.6, 0.67, 0.67, -0.75),
              doubleArrayOf(1.35, 1.52, 1.52, -1.69),
              doubleArrayOf(-0.33, -0.37, -0.37, 0.41),
              doubleArrayOf(0.78, 0.88, 0.88, -0.98),
              doubleArrayOf(1.18, 1.33, 1.33, -1.48)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.04, -0.08, 0.0, 0.07, -0.02),
              doubleArrayOf(-0.04, -0.08, 0.0, 0.07, -0.02),
              doubleArrayOf(0.2, 0.4, 0.0, -0.35, 0.09),
              doubleArrayOf(-0.28, -0.56, 0.0, 0.49, -0.13),
              doubleArrayOf(-0.08, -0.16, 0.0, 0.14, -0.04)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-2.65, -0.65, 1.59, 0.92)),
            tolerance = 0.005))
        }
        }
    }
  }
})
