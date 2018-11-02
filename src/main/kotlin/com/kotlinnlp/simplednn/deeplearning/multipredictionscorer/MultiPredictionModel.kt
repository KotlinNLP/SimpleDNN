/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The MultiPredictionModel contains the models of the sub-networks of a MultiPredictionScorer.
 *
 * @param networksConfig the list of configurations of the [FeedforwardNeuralNetwork]s
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class MultiPredictionModel(
  vararg networksConfig: MultiPredictionNetworkConfig,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [MultiPredictionModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [MultiPredictionModel]
     *
     * @return the [MultiPredictionModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): MultiPredictionModel = Serializer.deserialize(inputStream)
  }

  /**
   * The list of neural networks of this model.
   */
  val networks: List<NeuralNetwork> = List(
    size = networksConfig.size,
    init = { i ->
      FeedforwardNeuralNetwork(
        inputSize = networksConfig[i].inputSize,
        inputType = networksConfig[i].inputType,
        inputDropout = networksConfig[i].inputDropout,
        hiddenSize = networksConfig[i].hiddenSize,
        hiddenActivation = networksConfig[i].hiddenActivation,
        hiddenDropout = networksConfig[i].hiddenDropout,
        outputSize = networksConfig[i].outputSize,
        outputActivation = networksConfig[i].outputActivation,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer
      )
    }
  )

  /**
   * Serialize this [MultiPredictionModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [MultiPredictionModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
