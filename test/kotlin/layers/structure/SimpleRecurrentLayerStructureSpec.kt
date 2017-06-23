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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.59539, -0.8115, 0.17565, 0.88075, 0.08444)),
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
          val wContr: DenseNDArray = contributions.unit.weights.values as DenseNDArray
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
        layer.setInputRelevance(layerContributions = contributions)

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
            layer.setRecurrentRelevance(layerContributions = contributions)
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.59539, -0.8115, 0.17565, 0.88075, 0.08444)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributions") {
          val wContr: DenseNDArray = contributions.unit.weights.values as DenseNDArray
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
          val wRecContr: DenseNDArray = contributions.unit.recurrentWeights.values
          assertTrue {
            wRecContr.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.04, 0.1979, -0.19305, 0.7563, 0.50483),
                doubleArrayOf(0.13816, -0.15790, -0.05826, 0.50141, -0.46483),
                doubleArrayOf(0.14764, 0.14764, -0.23392, 0.32815, -0.36202),
                doubleArrayOf(0.08, 0.06026, -0.06566, 0.22326, 0.61123),
                doubleArrayOf(0.07843, 0.07843, -0.27305, 0.03163, 0.15921)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.setInputRelevance(layerContributions = contributions)
        layer.setRecurrentRelevance(layerContributions = contributions)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.27469, -0.95735, 0.85408, 1.65854)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.30048, 0.38969, -0.80763, 0.54242, 0.29448)),
              tolerance = 1.0e-05)
          }
        }
      }
    }

    context("backward") {

      on("without next and previous state") {

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
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.96693, -1.54688, 0.15, -0.93863, -0.63775)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
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
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
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

      on("with prev state only") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Back())
        val paramsErrors = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02539, -1.5615, 0.32565, -0.75925, -0.36556)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02539, -1.5615, 0.32565, -0.75925, -0.36556)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.02031, -0.02285, -0.02285, 0.02539),
              doubleArrayOf(1.2492, 1.40535, 1.40535, -1.5615),
              doubleArrayOf(-0.26052, -0.29308, -0.29308, 0.32565),
              doubleArrayOf(0.60740, 0.68333, 0.68333, -0.75925),
              doubleArrayOf(0.29245, 0.329, 0.329, -0.36556)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00501, 0.00501, -0.0074, -0.01819, -0.01686),
              doubleArrayOf(0.3082, -0.3082, 0.45489, 1.1185, 1.0369),
              doubleArrayOf(-0.06427, 0.06427, -0.09487, -0.23326, -0.21624),
              doubleArrayOf(0.14986, -0.14986, 0.22118, 0.54385, 0.50417),
              doubleArrayOf(0.07215, -0.07215, 0.10649, 0.26185, 0.24275)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.60603, 0.72965, 0.17712, 1.18027)),
            tolerance = 1.0e-05))
        }
      }

      on("with next state only") {

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
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.74789, -1.69287, 0.41, -0.97927, -1.47708)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
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
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-2.64621, -0.65432, 1.58598, 0.9243)),
            tolerance = 1.0e-05))
        }
      }

      on("with next and previous state") {

        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(SimpleRecurrentLayerContextWindow.Bilateral())
        val paramsErrors = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val outputGold = SimpleRecurrentLayerStructureUtils.getOutputGold()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(outputGold))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the output") {
          assertTrue(layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.19322, -1.69809, 0.57763, -0.77719, -1.22936)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.19322, -1.69809, 0.57763, -0.77719, -1.22936)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.15458, -0.1739, -0.1739, 0.19322),
              doubleArrayOf(1.35847, 1.52828, 1.52828, -1.69809),
              doubleArrayOf(-0.4621, -0.51986, -0.51986, 0.57763),
              doubleArrayOf(0.62176, 0.69947, 0.69947, -0.77719),
              doubleArrayOf(0.98349, 1.10642, 1.10642, -1.22936)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.03814, 0.03814, -0.05629, -0.1384, -0.12831),
              doubleArrayOf(0.33516, -0.33516, 0.49467, 1.21634, 1.12759),
              doubleArrayOf(-0.11401, 0.11401, -0.16827, -0.41375, -0.38356),
              doubleArrayOf(0.1534, -0.1534, 0.22641, 0.5567, 0.51609),
              doubleArrayOf(0.24264, -0.24264, 0.35813, 0.88059, 0.81634)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.80121, -0.13905, 0.70945, 0.62558)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
