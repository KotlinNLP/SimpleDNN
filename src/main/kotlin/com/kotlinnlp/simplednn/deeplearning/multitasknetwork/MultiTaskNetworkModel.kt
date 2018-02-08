/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multitasknetwork

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
 * The model of the [MultiTaskNetwork].
 *
 * @property inputSize the size of the input layer
 * @property inputType the type of the input array (default Dense)
 * @property inputDropout the probability of dropout of the input layer (default 0.0).
 *                        If applying it, the usual value is 0.25.
 * @property hiddenSize the size of the hidden layer
 * @property hiddenActivation the activation function of the hidden layer
 * @property hiddenDropout the probability of dropout of the hidden layer (default 0.0).
 *                         If applying it, the usual value is 0.5.
 * @property hiddenMeProp whether to use the 'meProp' errors propagation algorithm for the hidden layer (default false)
 * @property outputConfigurations a list of configurations of the output networks
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class MultiTaskNetworkModel(
  val inputSize: Int,
  val inputType: LayerType.Input = LayerType.Input.Dense,
  val inputDropout: Double = 0.0,
  val hiddenSize: Int,
  val hiddenActivation: ActivationFunction?,
  val hiddenDropout: Double = 0.0,
  val hiddenMeProp: Boolean = false,
  val outputConfigurations: List<MultiTaskNetworkConfig>,
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
     * Read a [MultiTaskNetworkModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [MultiTaskNetworkModel]
     *
     * @return the [MultiTaskNetworkModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): MultiTaskNetworkModel = Serializer.deserialize(inputStream)
  }

  /**
   * The input network (composed by a single layer).
   */
  val inputNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.inputSize,
      inputType = this.inputType,
      dropout = this.inputDropout),
    LayerConfiguration(
      size = this.hiddenSize,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = this.hiddenActivation,
      meProp = this.hiddenMeProp),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  /**
   * The list of output networks (each composed by a single layer).
   */
  val outputNetworks: List<NeuralNetwork> = this.outputConfigurations.map {
    NeuralNetwork(
      LayerConfiguration(
        size = this.hiddenSize,
        inputType = LayerType.Input.Dense,
        dropout = this.hiddenDropout),
      LayerConfiguration(
        size = it.outputSize,
        connectionType = LayerType.Connection.Feedforward,
        activationFunction = it.outputActivation,
        meProp = it.outputMeProp),
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
  }

  /**
   * The neural parameters of this model.
   */
  val params = MultiTaskNetworkParameters(
    inputParams = this.inputNetwork.model,
    outputParamsList = this.outputNetworks.map { it.model }
  )

  /**
   * Serialize this [MultiTaskNetworkModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [MultiTaskNetworkModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
