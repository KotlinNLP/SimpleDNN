/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning

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
 * The model of a [CLNetwork].
 *
 * @property classes the set of possible classes
 * @property inputSize the size of the input layer
 * @property hiddenSize the size of the hidden layer
 * @property hiddenActivation the activation function of the hidden layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class CLNetworkModel(
  val classes: Set<String>,
  val inputSize: Int,
  val hiddenSize: Int,
  val hiddenActivation: ActivationFunction?,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  /**
   * The map that associates each class to a feed-forward network.
   */
  val networks: Map<String, NeuralNetwork> = this.classes.associate { it to NeuralNetwork(

    LayerConfiguration(
      size = this.inputSize,
      inputType = LayerType.Input.Dense),

    LayerConfiguration(
      size = this.hiddenSize,
      activationFunction = this.hiddenActivation,
      connectionType = LayerType.Connection.Feedforward),

    LayerConfiguration(
      size = this.inputSize,
      activationFunction = null,
      connectionType = LayerType.Connection.Feedforward),

    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)
  }

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [CLNetworkModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [CLNetworkModel]
     *
     * @return the [CLNetworkModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): CLNetworkModel = Serializer.deserialize(inputStream)
  }

  /**
   * The parameters of all networks.
   */
  val params = CLNetworkParameters(this.networks.map { it.value.model })

  /**
   * Serialize this [CLNetworkModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [CLNetworkModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}