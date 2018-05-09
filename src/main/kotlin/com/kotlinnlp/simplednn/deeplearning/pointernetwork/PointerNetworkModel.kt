/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/


package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import java.io.Serializable


/**
 * The model of the [PointerNetwork].
 *
 * @property inputSize the size of the input vectors
 * @param recurrentHiddenSize the size of the hidden layer of the network used to encode to encode the input vectors
 * @param recurrentConnectionType the recurrent connection type of the network used to encode the input vectors
 * @param recurrentHiddenActivation the hidden activation function of the network used to encode the input vectors
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class PointerNetworkModel(
  val inputSize: Int,
  val recurrentHiddenSize: Int,
  private val recurrentHiddenActivation: ActivationFunction?,
  private val recurrentConnectionType: LayerType.Connection,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The recurrent network.
   */
  val recurrentNetwork = NeuralNetwork(
    LayerConfiguration(
      size = this.inputSize,
      inputType = LayerType.Input.Dense
    ),
    LayerConfiguration(
      size = recurrentHiddenSize,
      activationFunction = recurrentHiddenActivation,
      connectionType = recurrentConnectionType
    ),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  /**
   * The structure containing all the parameters of this model.
   */
  val params = PointerNetworkParameters(
    recurrentParams = this.recurrentNetwork.model)
}
