/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attention

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.deeplearning.attention.attentiverecurrentnetwork.AttentiveRecurrentNetwork
import com.kotlinnlp.simplednn.deeplearning.attention.attentiverecurrentnetwork.AttentiveRecurrentNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import deeplearning.attentiverecurrentnetwork.utils.AttentiveRecurrentNetworkUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 *
 */
class AttentiveRecurrentNetworkSpec : Spek({

  describe("an AttentiveRecurrentNetwork") {

    val utils = AttentiveRecurrentNetworkUtils

    on("forward") {

      val network = AttentiveRecurrentNetwork(model = utils.buildModel())
      val predictionLabels: List<DenseNDArray?> = utils.buildPredictionLabels()

      network.setInputSequence(utils.buildInputSequence1())

      val output0 = network.forward(lastPredictionLabel = predictionLabels[0], trainingMode = false)
      val output1 = network.forward(lastPredictionLabel = predictionLabels[1], trainingMode = false)
      val output2 = network.forward(lastPredictionLabel = predictionLabels[2], trainingMode = false)

      val expectedOutputs: List<DenseNDArray> = utils.buildExpectedOutputs1()

      it("should match the expected output at the forward n. 1") {
        assertTrue { output0.equals(expectedOutputs[0], tolerance = 1.0e-06) }
      }

      it("should match the expected output at the forward n. 2") {
        assertTrue { output1.equals(expectedOutputs[1], tolerance = 1.0e-06) }
      }

      it("should match the expected output at the forward n. 3") {
        assertTrue { output2.equals(expectedOutputs[2], tolerance = 1.0e-06) }
      }
    }

    on("forward with a second sequence") {

      val network = AttentiveRecurrentNetwork(model = utils.buildModel())
      val predictionLabels: List<DenseNDArray?> = utils.buildPredictionLabels()

      network.setInputSequence(utils.buildInputSequence1())

      network.forward(lastPredictionLabel = predictionLabels[0], trainingMode = false)
      network.forward(lastPredictionLabel = predictionLabels[1], trainingMode = false)
      network.forward(lastPredictionLabel = predictionLabels[2], trainingMode = false)

      network.setInputSequence(utils.buildInputSequence2())

      val output0 = network.forward(lastPredictionLabel = predictionLabels[0], trainingMode = false)
      val output1 = network.forward(lastPredictionLabel = predictionLabels[1], trainingMode = false)
      val output2 = network.forward(lastPredictionLabel = predictionLabels[2], trainingMode = false)

      val expectedOutputs: List<DenseNDArray> = utils.buildExpectedOutputs2()

      it("should match the expected output at the forward n. 1") {
        assertTrue { output0.equals(expectedOutputs[0], tolerance = 1.0e-06) }
      }

      it("should match the expected output at the forward n. 2") {
        assertTrue { output1.equals(expectedOutputs[1], tolerance = 1.0e-06) }
      }

      it("should match the expected output at the forward n. 3") {
        assertTrue { output2.equals(expectedOutputs[2], tolerance = 1.0e-06) }
      }
    }

    on("backward") {

      val network = AttentiveRecurrentNetwork(model = utils.buildModel())
      val predictionLabels: List<DenseNDArray?> = utils.buildPredictionLabels()

      network.setInputSequence(utils.buildInputSequence1())

      network.forward(lastPredictionLabel = predictionLabels[0], trainingMode = true)
      network.forward(lastPredictionLabel = predictionLabels[1], trainingMode = true)
      network.forward(lastPredictionLabel = predictionLabels[2], trainingMode = true)

      network.backward(outputErrors = utils.getOutputErrors1())

      val paramsErrors: AttentiveRecurrentNetworkParameters = network.getParamsErrors()
      val inputSequenceErrors: List<DenseNDArray> = network.getInputSequenceErrors()
      val contextLabelsErrors: List<DenseNDArray?> = network.getContextLabelsErrors()

      val expectedParamsErrors: AttentiveRecurrentNetworkParameters = utils.getExpectedParamsErrors1()
      val expectedInputErrors: List<DenseNDArray> = utils.getExpectedInputErrors1()
      val expectedLabelsErrors: List<DenseNDArray?> = utils.getExpectedLabelsErrors1()

      it("should match the expected errors of the transform layer weights") {
        assertTrue {
          paramsErrors.transformParams.unit.weights.values.equals(
            expectedParamsErrors.transformParams.unit.weights.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the transform layer biases") {
        assertTrue {
          paramsErrors.transformParams.unit.biases.values.equals(
            expectedParamsErrors.transformParams.unit.biases.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the context vector") {
        assertTrue {
          paramsErrors.attentionParams.attentionParams.contextVector.values.equals(
            expectedParamsErrors.attentionParams.attentionParams.contextVector.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the recurrent context network weights") {
        assertTrue {
          (paramsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
            .unit.weights.values.equals(
            (expectedParamsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
              .unit.weights.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the recurrent context network biases") {
        assertTrue {
          (paramsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
            .unit.biases.values.equals(
            (expectedParamsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
              .unit.biases.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the recurrent context network recurrent weights") {
        assertTrue {
          (paramsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
            .unit.recurrentWeights.values.equals(
            (expectedParamsErrors.recurrentContextParams.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
              .unit.recurrentWeights.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the output network weights") {
        assertTrue {
          (paramsErrors.outputParams.paramsPerLayer[0] as FeedforwardLayerParameters).unit.weights.values.equals(
            (expectedParamsErrors.outputParams.paramsPerLayer[0] as FeedforwardLayerParameters).unit.weights.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the output network biases") {
        assertTrue {
          (paramsErrors.outputParams.paramsPerLayer[0] as FeedforwardLayerParameters).unit.biases.values.equals(
            (expectedParamsErrors.outputParams.paramsPerLayer[0] as FeedforwardLayerParameters).unit.biases.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first context label") {
        assertNull(contextLabelsErrors[0])
      }

      it("should match the expected errors of the second context label") {
        assertTrue { contextLabelsErrors[1]!!.equals(expectedLabelsErrors[1]!!, tolerance = 1.0e-06) }
      }

      it("should match the expected errors of the third context label") {
        assertTrue { contextLabelsErrors[2]!!.equals(expectedLabelsErrors[2]!!, tolerance = 1.0e-06) }
      }

      it("should match the expected errors of the first array of the sequence") {
        assertTrue { inputSequenceErrors[0].equals(expectedInputErrors[0], tolerance = 1.0e-06) }
      }

      it("should match the expected errors of the second array of the sequence") {
        assertTrue { inputSequenceErrors[1].equals(expectedInputErrors[1], tolerance = 1.0e-06) }
      }

      it("should match the expected errors of the third array of the sequence") {
        assertTrue { inputSequenceErrors[2].equals(expectedInputErrors[2], tolerance = 1.0e-06) }
      }
    }
  }
})
