/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.on

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

    val processor = FeedforwardNeuralProcessor(MLP)

    val features = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))

    on("forward") {
      processor.forward(features)
    }
  }
})
