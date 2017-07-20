/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package layers.structure

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.recurrent.ran.RANLayerParameters
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
import layers.structure.contextwindows.RANLayerContextWindow
import layers.structure.utils.RANLayerStructureUtils
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class RANLayerStructureSpec : Spek({

  describe("a RANLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39652, 0.25162, 0.5, 0.70475, 0.45264)),
            tolerance = 1.0e-05))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.85321, 0.43291, 0.11609, 0.51999, 0.24232)),
            tolerance = 1.0e-05))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 1.0e-05))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.38375, -0.02516, 0.04996, 0.8918, -0.56369)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Back())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.72312, 0.24974, 0.54983, 0.82054, 0.53494)),
            tolerance = 1.0e-05))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91133, 0.18094, 0.04834, 0.67481, 0.38936)),
            tolerance = 1.0e-05))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 1.0e-05))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5045, 0.01121, 0.04046, 0.78504, -0.78786)),
            tolerance = 1.0e-05))
        }
      }
    }

    context("forward with relevance") {

      on("without previous state context") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Empty())
        val contributions = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward(layerContributions = contributions)

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39652, 0.25162, 0.5, 0.70475, 0.45264)),
            tolerance = 1.0e-05))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.85321, 0.43291, 0.11609, 0.51999, 0.24232)),
            tolerance = 1.0e-05))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 1.0e-05))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.38375, -0.02516, 0.04996, 0.8918, -0.56369)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributions of the input gate") {
          val inputGateContrib: DenseNDArray = contributions.inputGate.weights.values as DenseNDArray
          assertTrue {
            inputGateContrib.equals(
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

        it("should match the expected contributions of the candidate") {
          val candidateContrib: DenseNDArray = contributions.candidate.weights.values as DenseNDArray
          assertTrue {
            candidateContrib.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.85, -0.13, 0.05, 0.25),
                doubleArrayOf(0.56, -0.63, 0.27, -0.3),
                doubleArrayOf(-0.465, 0.315, -0.225, 0.475),
                doubleArrayOf(0.975, 0.715, -0.635, 0.975),
                doubleArrayOf(-0.475, -0.795, 0.735, -0.875)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(layerContributions = contributions)
        layer.setInputRelevance(layerContributions = contributions)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-6.80494, 7.20431, -4.37039, 4.97103)),
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

        val prevStateLayer = RANLayerContextWindow.Back().getPrevStateLayer()
        val contextWindow = mock<LayerContextWindow>()
        val layer = RANLayerStructureUtils.buildLayer(contextWindow)
        val contributions = RANLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevStateLayer()).thenReturn(prevStateLayer)

        layer.forward(layerContributions = contributions)

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.72312, 0.24974, 0.54983, 0.82054, 0.53494)),
            tolerance = 1.0e-05))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91133, 0.18094, 0.04834, 0.67481, 0.38936)),
            tolerance = 1.0e-05))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(1.02, -0.1, 0.1, 2.03, -1.41)),
            tolerance = 1.0e-05))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5045, 0.01121, 0.04046, 0.78504, -0.78786)),
            tolerance = 1.0e-05))
        }

        it("should match the expected contributions of the input gate") {
          val inputGateContrib: DenseNDArray = contributions.inputGate.weights.values as DenseNDArray
          assertTrue {
            inputGateContrib.equals(
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

        it("should match the expected contributions of the forget gate") {
          val forgetGateContrib: DenseNDArray = contributions.forgetGate.weights.values as DenseNDArray
          assertTrue {
            forgetGateContrib.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.0325, -0.2475, 1.0125, 0.5125),
                doubleArrayOf(-0.5350, 0.2050, -0.0650, 0.025),
                doubleArrayOf(-0.6725, -0.8325, 0.3375, -0.4125),
                doubleArrayOf(0.7450, -0.7850, 0.2950, -0.275),
                doubleArrayOf(0.4475, -0.6525, 0.4275, -0.9125)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected contributions of the candidate") {
          val candidateContrib: DenseNDArray = contributions.candidate.weights.values as DenseNDArray
          assertTrue {
            candidateContrib.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.85, -0.13, 0.05, 0.25),
                doubleArrayOf(0.56, -0.63, 0.27, -0.3),
                doubleArrayOf(-0.465, 0.315, -0.225, 0.475),
                doubleArrayOf(0.975, 0.715, -0.635, 0.975),
                doubleArrayOf(-0.475, -0.795, 0.735, -0.875)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected recurrent contributions of the input gate") {
          val inputGateContrib: DenseNDArray = contributions.inputGate.recurrentWeights.values
          assertTrue {
            inputGateContrib.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.04, 0.2, -0.2, 0.94, 0.60),
                doubleArrayOf(0.14, -0.16, -0.06, 0.63, -0.56),
                doubleArrayOf(0.15, 0.15, -0.24, 0.42, -0.43),
                doubleArrayOf(0.08, 0.06, -0.07, 0.26, 0.72),
                doubleArrayOf(0.08, 0.08, -0.28, 0.05, 0.20)
              )),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected recurrent contributions of the forget gate") {
          val forgetGateContrib: DenseNDArray = contributions.forgetGate.recurrentWeights.values
          assertTrue {
            forgetGateContrib.equals(
              DenseNDArrayFactory.arrayOf(arrayOf(
                doubleArrayOf(0.07, -0.03, 0.39, 0.18, 0.41),
                doubleArrayOf(-0.08, -0.16, 0.02, -0.7, -0.22),
                doubleArrayOf(-0.03, -0.27, -0.18, -0.99, 0.07),
                doubleArrayOf(-0.12, 0.06, -0.07, 0.38, 0.50),
                doubleArrayOf(-0.05, 0.01, -0.03, 0.72, -0.41)
              )),
              tolerance = 1.0e-05)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(layerContributions = contributions)
        layer.setInputRelevance(layerContributions = contributions)
        layer.setRecurrentRelevance(layerContributions = contributions)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected relevance of the input gate") {
          val relevance: DenseNDArray = layer.inputGate.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.13434, -0.09416, 0.16398, 0.15841, 0.07058)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected relevance of the forget gate") {
          val relevance: DenseNDArray = layer.forgetGate.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03434, 0.19416, -0.06398, -0.05841, 0.02942)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected relevance of the candidate") {
          val relevance: DenseNDArray = layer.candidate.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.13434, -0.09416, 0.16398, 0.15841, 0.07058)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.73699, 0.21761, -0.13861, 1.13203)),
              tolerance = 1.0e-05)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.15578, 0.3737, -0.40348, 0.45246, -0.05248)),
              tolerance = 1.0e-05)
          }
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Empty())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.15882, -0.77467, 0.19946, -0.15316, -0.69159)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03877, 0.01459, 0.00499, -0.06469, 0.2416)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06298, -0.19492, 0.09973, -0.10794, -0.31304)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03877, 0.01459, 0.00499, -0.06469, 0.2416)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06298, -0.19492, 0.09973, -0.10794, -0.31304)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.03101, 0.03489, 0.03489, -0.03877),
              doubleArrayOf(-0.01167, -0.01313, -0.01313, 0.01459),
              doubleArrayOf(-0.00399, -0.00449, -0.00449, 0.00499),
              doubleArrayOf(0.05175, 0.05822, 0.05822, -0.06469),
              doubleArrayOf(-0.19328, -0.21744, -0.21744, 0.2416)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.05038, 0.05668, 0.05668, -0.06298),
              doubleArrayOf(0.15594, 0.17543, 0.17543, -0.19492),
              doubleArrayOf(-0.07978, -0.08976, -0.08976, 0.09973),
              doubleArrayOf(0.08635, 0.09714, 0.09714, -0.10794),
              doubleArrayOf(0.25044, 0.28174, 0.28174, -0.31304)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.21996, -0.12731, 0.10792, 0.49361)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous state only") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Back())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04883, -0.73869, 0.19015, -0.32806, -0.46949)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.00997, 0.01384, 0.00471, -0.09807, 0.16469)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00078, -0.02161, -0.00255, 0.05157, 0.07412)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03531, -0.18448, 0.10455, -0.26919, -0.25115)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.00997, 0.01384, 0.00471, -0.09807, 0.16469)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00078, -0.02161, -0.00255, 0.05157, 0.07412)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03531, -0.18448, 0.10455, -0.26919, -0.25115)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.00798, 0.00898, 0.00898, -0.00997),
              doubleArrayOf(-0.01107, -0.01246, -0.01246, 0.01384),
              doubleArrayOf(-0.00377, -0.00424, -0.00424, 0.00471),
              doubleArrayOf(0.07845, 0.08826, 0.08826, -0.09807),
              doubleArrayOf(-0.13175, -0.14822, -0.14822, 0.16469)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00062, -0.0007, -0.0007, 0.00078),
              doubleArrayOf(0.01729, 0.01945, 0.01945, -0.02161),
              doubleArrayOf(0.00204, 0.00229, 0.00229, -0.00255),
              doubleArrayOf(-0.04125, -0.04641, -0.04641, 0.05157),
              doubleArrayOf(-0.0593, -0.06671, -0.06671, 0.07412)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.02825, 0.03178, 0.03178, -0.03531),
              doubleArrayOf(0.14759, 0.16603, 0.16603, -0.18448),
              doubleArrayOf(-0.08364, -0.09409, -0.09409, 0.10455),
              doubleArrayOf(0.21535, 0.24227, 0.24227, -0.26919),
              doubleArrayOf(0.20092, 0.22604, 0.22604, -0.25115)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.00199, -0.00199, 0.00299, 0.00898, 0.00798),
              doubleArrayOf(-0.00277, 0.00277, -0.00415, -0.01246, -0.01107),
              doubleArrayOf(-0.00094, 0.00094, -0.00141, -0.00424, -0.00377),
              doubleArrayOf(0.01961, -0.01961, 0.02942, 0.08826, 0.07845),
              doubleArrayOf(-0.03294, 0.03294, -0.04941, -0.14822, -0.13175)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00016, 0.00016, -0.00023, -0.0007, -0.00062),
              doubleArrayOf(0.00432, -0.00432, 0.00648, 0.01945, 0.01729),
              doubleArrayOf(0.00051, -0.00051, 0.00076, 0.00229, 0.00204),
              doubleArrayOf(-0.01031, 0.01031, -0.01547, -0.04641, -0.04125),
              doubleArrayOf(-0.01482, 0.01482, -0.02224, -0.06671, -0.0593)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.21972, 0.09327, -0.127, 0.17217)),
            tolerance = 1.0e-05))
        }
      }

      on("with next state only") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Front())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.30882, -0.42467, 0.59946, -1.00316, -0.88159)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.07538, 0.008, 0.01499, -0.42373, 0.30797)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.12245, -0.10685, 0.29973, -0.70697, -0.39905)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.07538, 0.008, 0.01499, -0.42373, 0.30797)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.12245, -0.10685, 0.29973, -0.70697, -0.39905)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0603, 0.06784, 0.06784, -0.07538),
              doubleArrayOf(-0.0064, -0.0072, -0.0072, 0.00800),
              doubleArrayOf(-0.01199, -0.01349, -0.01349, 0.01499),
              doubleArrayOf(0.33899, 0.38136, 0.38136, -0.42373),
              doubleArrayOf(-0.24638, -0.27718, -0.27718, 0.30797)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.09796, 0.11021, 0.11021, -0.12245),
              doubleArrayOf(0.08548, 0.09617, 0.09617, -0.10685),
              doubleArrayOf(-0.23978, -0.26976, -0.26976, 0.29973),
              doubleArrayOf(0.56558, 0.63627, 0.63627, -0.70697),
              doubleArrayOf(0.31924, 0.35914, 0.35914, -0.39905)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.55722, 0.45624, -0.39506, 0.30611)),
            tolerance = 1.0e-05))
        }
      }

      on("with previous and next state") {

        val layer = RANLayerStructureUtils.buildLayer(RANLayerContextWindow.Bilateral())
        val paramsErrors = RANLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = RANLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.19883, -0.38869, 0.59015, -1.17806, -0.65949)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04061, 0.00728, 0.01461, -0.35216, 0.23134)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00317, -0.01137, -0.00791, 0.18518, 0.10412)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.14378, -0.09707, 0.32448, -0.96664, -0.35279)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.04061, 0.00728, 0.01461, -0.35216, 0.23134)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.00317, -0.01137, -0.00791, 0.18518, 0.10412)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.14378, -0.09707, 0.32448, -0.96664, -0.35279)),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.03248, 0.03655, 0.03655, -0.04061),
              doubleArrayOf(-0.00583, -0.00655, -0.00655, 0.00728),
              doubleArrayOf(-0.01169, -0.01315, -0.01315, 0.01461),
              doubleArrayOf(0.28172, 0.31694, 0.31694, -0.35216),
              doubleArrayOf(-0.18507, -0.20820, -0.20820, 0.23134)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00254, -0.00285, -0.00285, 0.00317),
              doubleArrayOf(0.00910, 0.01023, 0.01023, -0.01137),
              doubleArrayOf(0.00633, 0.00712, 0.00712, -0.00791),
              doubleArrayOf(-0.14814, -0.16666, -0.16666, 0.18518),
              doubleArrayOf(-0.08330, -0.09371, -0.09371, 0.10412)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.11502, 0.12940, 0.12940, -0.14378),
              doubleArrayOf(0.07766, 0.08737, 0.08737, -0.09707),
              doubleArrayOf(-0.25959, -0.29204, -0.29204, 0.32448),
              doubleArrayOf(0.77332, 0.86998, 0.86998, -0.96664),
              doubleArrayOf(0.28223, 0.31751, 0.31751, -0.35279)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(0.00812, -0.00812, 0.01218, 0.03655, 0.03248),
              doubleArrayOf(-0.00146, 0.00146, -0.00218, -0.00655, -0.00583),
              doubleArrayOf(-0.00292, 0.00292, -0.00438, -0.01315, -0.01169),
              doubleArrayOf(0.07043, -0.07043, 0.10565, 0.31694, 0.28172),
              doubleArrayOf(-0.04627, 0.04627, -0.06940, -0.20820, -0.18507)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              doubleArrayOf(-0.00063, 0.00063, -0.00095, -0.00285, -0.00254),
              doubleArrayOf(0.00227, -0.00227, 0.00341, 0.01023, 0.00910),
              doubleArrayOf(0.00158, -0.00158, 0.00237, 0.00712, 0.00633),
              doubleArrayOf(-0.03704, 0.03704, -0.05555, -0.16666, -0.14814),
              doubleArrayOf(-0.02082, 0.02082, -0.03124, -0.09371, -0.08330)
            )),
            tolerance = 1.0e-05))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.65243, 0.74348, -0.76607, -0.15266)),
            tolerance = 1.0e-05))
        }
      }
    }
  }
})
