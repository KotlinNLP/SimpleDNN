/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.encoders.sequenceencoder

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * Sequence Feedforward Network.
 *
 * It encodes a sequence of arrays into another sequence of arrays through a single Feedforward layer.
 *
 * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
 * @property inputSize the size of the input layer
 * @property outputSize the size of the output layer
 * @property outputActivation the activation function of the output layer (could be null)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class SequenceFeedforwardNetwork(
  val inputType: LayerType.Input,
  val inputSize: Int,
  val outputSize: Int,
  val outputActivation: ActivationFunction?,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [SequenceFeedforwardNetwork] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [SequenceFeedforwardNetwork]
     *
     * @return the [SequenceFeedforwardNetwork] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): SequenceFeedforwardNetwork = Serializer.deserialize(inputStream)
  }

  /**
   * The Feedforward Neural Network which encodes each input array into another array.
   */
  val network: NeuralNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.inputSize,
      inputType = this.inputType),
    LayerConfiguration(
      size = this.outputSize,
      activationFunction = this.outputActivation,
      connectionType = LayerType.Connection.Feedforward
    ),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  /**
   * Serialize this [SequenceFeedforwardNetwork] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [SequenceFeedforwardNetwork]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
