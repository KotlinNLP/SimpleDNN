/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class SimpleRecurrentLayerStructureSpec : Spek({

  describe("a SimpleRecurrentLayer") {

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
              DenseNDArrayFactory.arrayOf(listOf(
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

        val prevStateLayer = SimpleRecurrentLayerContextWindow.Back().getPrevState()
        val contextWindow = mock<LayerContextWindow>()
        val layer = SimpleRecurrentLayerStructureUtils.buildLayer(contextWindow)
        val contributions = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevState()).thenReturn(prevStateLayer)

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
              DenseNDArrayFactory.arrayOf(listOf(
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
          val wRecContr: DenseNDArray = contributions.unit.recurrentWeights.values as DenseNDArray
          assertTrue {
            wRecContr.equals(
              DenseNDArrayFactory.arrayOf(listOf(
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81459, -0.56459, 0.15, -0.47689, -0.61527)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.81459, -0.56459, 0.15, -0.47689, -0.61527)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.65167, 0.73313, 0.73313, -0.81459),
              doubleArrayOf(0.45167, 0.50813, 0.50813, -0.56459),
              doubleArrayOf(-0.12, -0.135, -0.135, 0.15),
              doubleArrayOf(0.38151, 0.4292, 0.4292, -0.47689),
              doubleArrayOf(0.49221, 0.55374, 0.55374, -0.61527)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.32512, -0.55398, 1.0709, 0.5709)),
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01639, -0.53319, 0.3156, -0.17029, -0.36295)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01639, -0.53319, 0.3156, -0.17029, -0.36295)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.01311, -0.01475, -0.01475, 0.01639),
              doubleArrayOf(0.42655, 0.47987, 0.47987, -0.53319),
              doubleArrayOf(-0.25248, -0.28404, -0.28404, 0.3156),
              doubleArrayOf(0.13623, 0.15326, 0.15326, -0.17029),
              doubleArrayOf(0.29036, 0.32666, 0.32666, -0.36295)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.00323, 0.00323, -0.00477, -0.01174, -0.01088),
              doubleArrayOf(0.10524, -0.10524, 0.15533, 0.38193, 0.35406),
              doubleArrayOf(-0.06229, 0.06229, -0.09194, -0.22606, -0.20957),
              doubleArrayOf(0.03361, -0.03361, 0.04961, 0.12198, 0.11308),
              doubleArrayOf(0.07164, -0.07164, 0.10573, 0.25998, 0.24101)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.42553, -0.20751, 0.28232, 0.30119)),
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.59555, -0.71058, 0.41, -0.51754, -1.4546)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.59555, -0.71058, 0.41, -0.51754, -1.4546)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.47644, 0.536, 0.536, -0.59555),
              doubleArrayOf(0.56847, 0.63952, 0.63952, -0.71058),
              doubleArrayOf(-0.328, -0.369, -0.369, 0.41),
              doubleArrayOf(0.41403, 0.46578, 0.46578, -0.51754),
              doubleArrayOf(1.16368, 1.30914, 1.30914, -1.4546)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.50405, -1.34891, 1.5466, 0.01887)),
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
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18422, -0.66978, 0.56758, -0.18823, -1.22675)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the biases") {
          assertTrue(paramsErrors.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.18422, -0.66978, 0.56758, -0.18823, -1.22675)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the weights") {
          assertTrue((paramsErrors.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.14738, -0.1658, -0.1658, 0.18422),
              doubleArrayOf(0.53582, 0.60280, 0.60280, -0.66978),
              doubleArrayOf(-0.45406, -0.51082, -0.51082, 0.56758),
              doubleArrayOf(0.15058, 0.16941, 0.16941, -0.18823),
              doubleArrayOf(0.9814, 1.10408, 1.10408, -1.22675)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the recurrent weights") {
          assertTrue(paramsErrors.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.03636, 0.03636, -0.05367, -0.13196, -0.12233),
              doubleArrayOf(0.1322, -0.1322, 0.19511, 0.47976, 0.44476),
              doubleArrayOf(-0.11203, 0.11203, -0.16534, -0.40656, -0.37689),
              doubleArrayOf(0.03715, -0.03715, 0.05483, 0.13483, 0.12499),
              doubleArrayOf(0.24213, -0.24213, 0.35737, 0.87872, 0.81461)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input") {
          assertTrue(layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.62071, -1.07621, 0.81464, -0.2535)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
