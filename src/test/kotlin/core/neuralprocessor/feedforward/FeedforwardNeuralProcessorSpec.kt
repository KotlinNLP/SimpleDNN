/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralprocessor.feedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe

/**
 *
 */
class FeedforwardNeuralProcessorSpec : Spek({

  describe("a FeedForwardLayerStructure") {

    val MLP = FeedforwardNeuralNetwork(
      inputSize = 4,
      hiddenSize = 5,
      hiddenActivation = Tanh(),
      outputSize = 3,
      outputActivation = Softmax())

    // TODO: set fixed params

    val processor = FeedforwardNeuralProcessor<DenseNDArray>(MLP, useDropout = false, propagateToInput = false)

    val features = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))

    context("forward") {
      processor.forward(features)
    }
  }
})
