/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.feedforward.simple.FeedforwardLayerStructureUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class FeedforwardNeuralProcessorSpec : Spek({

  describe("a FeedforwardNeuralProcessor") {

    val network = FeedforwardNeuralNetwork(
      inputSize = 4,
      hiddenSize = 5,
      hiddenActivation = Tanh,
      outputSize = 3,
      outputActivation = null
    ).apply {

      getLayerParams<FeedforwardLayerParameters>(0).apply {

        val inputParams: FeedforwardLayerParameters = FeedforwardLayerStructureUtils.getParams45()

        unit.weights.values.assignValues(inputParams.unit.weights.values)
        unit.biases.values.assignValues(inputParams.unit.biases.values)
      }

      getLayerParams<FeedforwardLayerParameters>(1).apply {

        val outputParams: FeedforwardLayerParameters = FeedforwardLayerStructureUtils.getParams53()

        unit.weights.values.assignValues(outputParams.unit.weights.values)
        unit.biases.values.assignValues(outputParams.unit.biases.values)
      }
    }

    context("forward") {

      val processor = FeedforwardNeuralProcessor<DenseNDArray>(network, propagateToInput = false)
      val input: DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))

      processor.forward(input)

      it("should match the expected output values") {

        assertTrue {
          processor.getOutput().equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.862641, -0.427188, 0.183369)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val processor = FeedforwardNeuralProcessor<DenseNDArray>(network, propagateToInput = true)
      val input: DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))
      val outputErrors: DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.4, -0.3))

      processor.forward(input)
      processor.backward(outputErrors)

      it("should match the expected input errors") {

        assertTrue {
          processor.getInputErrors().equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.811233, -0.182174, 0.417836, -0.291757)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
