/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
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

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.79688, 0.0, 0.70137, -0.18775)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Back())
        layer.forward()

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.74428, -0.8005, 0.19738, 0.9087, 0.13909)),
            tolerance = 1.0e-05))
        }
      }
    }

    context("forward with relevance") {

      on("without previous state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Empty())
        val contributions = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward(layerContributions = contributions)

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.39693, -0.79688, 0.0, 0.70137, -0.18775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.weights.values as DenseNDArray
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
        layer.calculateInputRelevance(layerContributions = contributions)

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
            layer.calculateRecurrentRelevance(layerContributions = contributions)
          }
        }
      }

      on("with previous state context") {

        val prevStateLayer = SimpleRecurrentLayerContextWindow.Back().getPrevStateLayer()
        val contextWindow = mock<LayerContextWindow>()
        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(contextWindow)
        val contributions = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevStateLayer()).thenReturn(prevStateLayer)

        layer.forward(layerContributions = contributions)

        it("should match the expected output") {
          assertTrue(layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.74428, -0.8005, 0.19738, 0.9087, 0.13909)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.weights.values as DenseNDArray
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

        it("should match the expected recurrent contributions") {
          val wRecContr: DenseNDArray = contributions.recurrentWeights.values
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
        layer.calculateInputRelevance(layerContributions = contributions)
        layer.calculateRecurrentRelevance(layerContributions = contributions)

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

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.96693, -1.54688, 0.15, -0.93863, -0.63775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.96693, -1.54688, 0.15, -0.93863, -0.63775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.77354, 0.87024, 0.87024, -0.96693),
              doubleArrayOf(1.2375, 1.39219, 1.39219, -1.54688),
              doubleArrayOf(-0.12, -0.135, -0.135, 0.15),
              doubleArrayOf(0.7509, 0.84476, 0.84476, -0.93863),
              doubleArrayOf(0.5102, 0.57397, 0.57397, -0.63775)
            )),
            tolerance = 1.0e-05))
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
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-2.46728, 0.14061, 1.11028, 1.47633)),
            tolerance = 1.0e-05))
        }
      }

      on("with next state context") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Front())
        val paramsErrors = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.74789, -1.69287, 0.41, -0.97927, -1.47708)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.74789, -1.69287, 0.41, -0.97927, -1.47708)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.59832, 0.6731, 0.67310, -0.74789),
              doubleArrayOf(1.3543, 1.52359, 1.52359, -1.69287),
              doubleArrayOf(-0.328, -0.369, -0.369, 0.41),
              doubleArrayOf(0.78342, 0.88134, 0.88134, -0.97927),
              doubleArrayOf(1.18166, 1.32937, 1.32937, -1.47708)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.03969, -0.07969, 0.0, 0.07014, -0.01877),
              doubleArrayOf(-0.03969, -0.07969, 0.0, 0.07014, -0.01877),
              doubleArrayOf(0.19847, 0.39844, 0.0, -0.35069, 0.09387),
              doubleArrayOf(-0.27785, -0.55781, 0.0, 0.49096, -0.13142),
              doubleArrayOf(-0.07939, -0.15938, 0.0, 0.14027, -0.03755)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-2.64621, -0.65432, 1.58598, 0.9243)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
