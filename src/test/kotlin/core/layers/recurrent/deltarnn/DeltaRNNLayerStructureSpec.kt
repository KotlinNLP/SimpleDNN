/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayerParameters
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
class DeltaRNNLayerStructureSpec : Spek({

  describe("a DeltaRNNLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.568971, 0.410323, -0.39693, 0.648091, -0.449441)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.519989, 0.169384, 0.668188, 0.325195, 0.601088)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.287518, 0.06939, -0.259175, 0.20769, -0.263768)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Back())
        layer.forward()

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.576403, 0.40594, -0.222741, 0.36182, -0.42253)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.519989, 0.169384, 0.668188, 0.325195, 0.601088)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.202158, 0.228591, -0.240679, -0.350224, -0.476828)),
              tolerance = 1.0e-06)
          }
        }
      }
    }

    context("forward with relevance") {

      on("without previous state context") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Empty())
        val contributions = DeltaRNNLayerParameters(
          inputSize = 4,
          outputSize = 5,
          weightsInitializer = null,
          biasesInitializer = null)

        layer.forward(layerContributions = contributions)

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.568971, 0.410323, -0.39693, 0.648091, -0.449441)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.519989, 0.169384, 0.668188, 0.325195, 0.601088)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.287518, 0.06939, -0.259175, 0.20769, -0.263768)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected contributions of the input") {
          val inputContrib: DenseNDArray = contributions.feedforwardUnit.weights.values as DenseNDArray
          assertTrue {
            inputContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.4, -0.54, 0.72, -0.6),
                doubleArrayOf(-0.56, 0.36, -0.09, -0.8),
                doubleArrayOf(-0.56, 0.63, -0.27, 0.5),
                doubleArrayOf(-0.64, 0.81, 0.0, -0.1),
                doubleArrayOf(-0.32, -0.9, 0.63, 0.8)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected recurrent contributions") {
          val recContrib: DenseNDArray = contributions.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            recContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 1.0e-06)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(layerContributions = contributions)
        layer.setInputRelevance(layerContributions = contributions)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected relevance of the partition array") {
          val relevance: DenseNDArray = layer.partition.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, 0.1, 0.1, 0.1)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected relevance of the candidate") {
          val relevance: DenseNDArray = layer.candidate.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, 0.1, 0.1, 0.1)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected relevance of the d1 input array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d1Input.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, 0.1, 0.1, 0.1)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.058871, -0.524906, 1.314539, 0.269238)),
              tolerance = 1.0e-06)
          }
        }

        it("should throw an Exception when calculating the recurrent relevance") {
          assertFailsWith <KotlinNullPointerException> {
            layer.setRecurrentRelevance(layerContributions = contributions)
          }
        }
      }

      on("with previous state context") {

        val prevStateLayer = DeltaLayerContextWindow.Back().getPrevStateLayer()
        val contextWindow = mock<LayerContextWindow>()
        val layer = DeltaRNNLayerStructureUtils.buildLayer(contextWindow)
        val contributions = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        whenever(contextWindow.getPrevStateLayer()).thenReturn(prevStateLayer)

        layer.forward(layerContributions = contributions)

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.576403, 0.40594, -0.222741, 0.36182, -0.42253)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected partition array") {
          assertTrue {
            layer.partition.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.519989, 0.169384, 0.668188, 0.325195, 0.601088)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected output") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.202158, 0.228591, -0.240679, -0.350224, -0.476828)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected contributions of the input") {
          val inputContrib: DenseNDArray = contributions.feedforwardUnit.weights.values as DenseNDArray
          assertTrue {
            inputContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.4, -0.54, 0.72, -0.6),
                doubleArrayOf(-0.56, 0.36, -0.09, -0.8),
                doubleArrayOf(-0.56, 0.63, -0.27, 0.5),
                doubleArrayOf(-0.64, 0.81, 0.0, -0.1),
                doubleArrayOf(-0.32, -0.9, 0.63, 0.8)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected recurrent contributions") {
          val recContrib: DenseNDArray = contributions.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            recContrib.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.1579, -0.23305, 0.716298, 0.464826),
                doubleArrayOf(0.138163, -0.1579, -0.058263, 0.501409, -0.464826),
                doubleArrayOf(0.177638, 0.177638, -0.203919, 0.358149, -0.332018),
                doubleArrayOf(0.0, -0.019738, -0.145656, 0.14326, 0.531229),
                doubleArrayOf(0.118425, 0.118425, -0.23305, 0.07163, 0.199211)
              )),
              tolerance = 1.0e-06)
          }
        }

        layer.setOutputRelevance(DistributionArray.uniform(length = 5))
        layer.propagateRelevanceToGates(layerContributions = contributions)
        layer.setInputRelevance(layerContributions = contributions)
        layer.setRecurrentRelevance(layerContributions = contributions)

        it("should set a Dense input relevance") {
          assertTrue { layer.inputArray.relevance is DenseNDArray }
        }

        it("should match the expected relevance of the partition array") {
          val relevance: DenseNDArray = layer.partition.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, 0.1, 0.1, 0.1)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected relevance of the candidate") {
          val relevance: DenseNDArray = layer.candidate.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.151155, 0.030391, 0.06021, -0.029987, 0.048968)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected relevance of the d1 input array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d1Input.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.101818, 0.031252, 0.074148, -0.028935, 0.031181)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected relevance of the d1 recurrent array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d1Rec.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.05417, 0.000358, -0.00857, 0.000304, 0.018798)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected relevance of the d2 array") {
          val relevance: DenseNDArray = layer.relevanceSupport.d2.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.103506, -0.001219, -0.005368, -0.001356, -0.001011)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input relevance") {
          val relevance: DenseNDArray = layer.inputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.156729, -0.419269, 1.161411, 0.171329)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected recurrent relevance") {
          val relevance: DenseNDArray = prevStateLayer.outputArray.relevance as DenseNDArray
          assertTrue {
            relevance.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.077057, 0.03487, 0.116601, 0.037238, 0.131607)),
              tolerance = 1.0e-06)
          }
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Empty())
        val paramsErrors = DeltaRNNLayerParameters(
          inputSize = 4,
          outputSize = 5,
          weightsInitializer = null,
          biasesInitializer = null)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.25913, -0.677332, -0.101842, -1.370527, -0.664109)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.091124, -0.095413, -0.057328, -0.258489, -0.318553)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.0368, -0.039102, 0.008963, -0.194915, 0.071569)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.091124, -0.095413, -0.057328, -0.258489, -0.318553)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.0368, -0.039102, 0.008963, -0.194915, 0.071569)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.074722, 0.104, -0.017198, -0.018094, -0.066896)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.007571, 0.008517, 0.008517, -0.009463),
                doubleArrayOf(0.00075, 0.000843, 0.000843, -0.000937),
                doubleArrayOf(-0.025515, -0.028704, -0.028704, 0.031894),
                doubleArrayOf(0.073215, 0.082367, 0.082367, -0.091519),
                doubleArrayOf(-0.159192, -0.179091, -0.179091, 0.19899)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.023319, 0.253729, -0.122248, 0.190719)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous state only") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Back())
        val paramsErrors = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.352809, -0.494163, -0.085426, -1.746109, -0.7161)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.122505, -0.06991, -0.054249, -0.493489, -0.353592)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06814, -0.0145, -0.001299, -0.413104, -0.041468)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.122505, -0.06991, -0.054249, -0.493489, -0.353592)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06814, -0.0145, -0.001299, -0.413104, -0.041468)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1111, -0.003156, -0.002889, -0.017586, -0.020393)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.100454, 0.076202, -0.016275, -0.034544, -0.074254)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.135487, 0.002895, -0.009628, -0.251233, -0.097111)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.029084, -0.03272, -0.03272, 0.036355),
                doubleArrayOf(-0.010076, -0.011335, -0.011335, 0.012595),
                doubleArrayOf(-0.01401, -0.015761, -0.015761, 0.017512),
                doubleArrayOf(0.252961, 0.284582, 0.284582, -0.316202),
                doubleArrayOf(-0.072206, -0.081232, -0.081232, 0.090257)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.000242, -0.000242, 0.000357, 0.000878, 0.000813),
                doubleArrayOf(0.001752, -0.001752, 0.002586, 0.00636, 0.005896),
                doubleArrayOf(0.011671, -0.011671, 0.017226, 0.042355, 0.039265),
                doubleArrayOf(-0.075195, 0.075195, -0.110982, -0.272891, -0.252981),
                doubleArrayOf(0.008445, -0.008445, 0.012464, 0.030647, 0.028411)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.177606, 0.379355, -0.085751, 0.080693)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with next state only") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Front())
        val paramsErrors = DeltaRNNLayerParameters(
          inputSize = 4,
          outputSize = 5,
          weightsInitializer = null,
          biasesInitializer = null)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.075296, -0.403656, -0.191953, -0.383426, -0.699093)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.026478, -0.056861, -0.108053, -0.072316, -0.335333)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.010693, -0.023303, 0.016893, -0.05453, 0.07534)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.026478, -0.056861, -0.108053, -0.072316, -0.335333)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.010693, -0.023303, 0.016893, -0.05453, 0.07534)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.021712, 0.061979, -0.032416, -0.005062, -0.07042)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0022, 0.002475, 0.002475, -0.00275),
                doubleArrayOf(0.000447, 0.000503, 0.000503, -0.000558),
                doubleArrayOf(-0.048091, -0.054102, -0.054102, 0.060114),
                doubleArrayOf(0.020483, 0.023044, 0.023044, -0.025604),
                doubleArrayOf(-0.167578, -0.188526, -0.188526, 0.209473)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.10362, 0.18901, -0.126453, 0.202292)),
              tolerance = 1.0e-06)
          }
        }
      }

      on("with previous and next state") {

        val layer = DeltaRNNLayerStructureUtils.buildLayer(DeltaLayerContextWindow.Bilateral())
        val paramsErrors = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        layer.outputArray.assignErrors(layer.outputArray.values.sub(DeltaRNNLayerStructureUtils.getOutputGold()))
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.160599, -0.233533, -0.17643, -0.841042, -0.745151)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate array") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.055764, -0.033038, -0.11204, -0.237697, -0.367937)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition array") {
          assertTrue {
            layer.partition.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.031017, -0.006853, -0.002682, -0.198978, -0.043151)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the candidate biases") {
          assertTrue {
            paramsErrors.feedforwardUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.055764, -0.033038, -0.11204, -0.237697, -0.367937)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the partition biases") {
          assertTrue {
            paramsErrors.recurrentUnit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.031017, -0.006853, -0.002682, -0.198978, -0.043151)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the alpha array") {
          assertTrue {
            paramsErrors.alpha.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.050573, -0.001492, -0.005966, -0.008471, -0.021221)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta1 array") {
          assertTrue {
            paramsErrors.beta1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.045727, 0.036012, -0.033612, -0.016639, -0.077267)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the beta2 array") {
          assertTrue {
            paramsErrors.beta2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.061674, 0.001368, -0.019886, -0.121011, -0.10105)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the weights") {
          assertTrue {
            (paramsErrors.feedforwardUnit.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.013239, -0.014894, -0.014894, 0.016549),
                doubleArrayOf(-0.004762, -0.005357, -0.005357, 0.005952),
                doubleArrayOf(-0.028934, -0.032551, -0.032551, 0.036168),
                doubleArrayOf(0.121843, 0.137073, 0.137073, -0.152304),
                doubleArrayOf(-0.075135, -0.084527, -0.084527, 0.093919)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the recurrent weights") {
          val wRec: DenseNDArray = paramsErrors.recurrentUnit.weights.values as DenseNDArray
          assertTrue {
            wRec.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.00011, -0.00011, 0.000162, 0.000399, 0.000370),
                doubleArrayOf(0.000828, -0.000828, 0.001222, 0.003005, 0.002786),
                doubleArrayOf(0.024104, -0.024104, 0.035576, 0.087477, 0.081094),
                doubleArrayOf(-0.036219, 0.036219, -0.053457, -0.131442, -0.121852),
                doubleArrayOf(0.008787, -0.008787, 0.012969, 0.03189, 0.029563)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.046517, 0.213223, -0.067537, 0.093758)),
              tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
