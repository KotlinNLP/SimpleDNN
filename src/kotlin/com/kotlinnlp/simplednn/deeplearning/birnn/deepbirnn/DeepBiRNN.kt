/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN

/**
 * The DeepBiRNN.
 *
 * A deep bidirectional RNN (or k-layer BiRNN) is composed of k BiRNN
 * BiRNN(1), ..., BiRNN(k) that feed into each other: the output of BIRNN(i)
 * becomes the input of BiRNN(i+1). Stacking BiRNNs in this way has been empirically
 * shown to be effective (Irsoy and Cardie, 2014).
 *
 * @property inputType the type of the input layer (Dense, Sparse, SparseBinary)
 * @property inputSize the size of the input layer
 * @property hiddenActivation the activation function of the hidden layer
 * @property recurrentConnectionType type of recurrent neural network (e.g. LSTM, GRU, CFN, SimpleRNN)
 * @property numberOfLayers number of stacked BiRNNs
 *
 * Note: each BiRNN has the hidden layer of the same size as of its input.
 */
class DeepBiRNN(val inputType: LayerType.Input,
                val inputSize: Int,
                val hiddenActivation: ActivationFunction?,
                val recurrentConnectionType: LayerType.Connection,
                val numberOfLayers: Int = 1){

  /**
   * Stacked BiRNNs
   */
  val layers: Array<BiRNN>

  /**
   * The size of the output layer (of the last BiRNN)
   */
  val outputSize: Int get() = layers.last().outputSize

  init {
    require(this.numberOfLayers > 0) {
      "required at least one BiRNN layer"
    }

    require(this.recurrentConnectionType.property == LayerType.Property.Recurrent) {
      "required recurrentConnectionType with Recurrent property"
    }

    val initLayers = arrayOfNulls<BiRNN>(size = this.numberOfLayers)

    for (i in 0 until this.numberOfLayers){

      initLayers[i] = if (i == 0) {

        BiRNN(
          inputSize = this.inputSize,
          hiddenSize = this.inputSize,
          hiddenActivation = this.hiddenActivation,
          recurrentConnectionType = this.recurrentConnectionType,
          inputType = this.inputType)

      } else {

        BiRNN(
          inputSize = initLayers[i - 1]!!.outputSize,
          hiddenSize = initLayers[i - 1]!!.outputSize,
          hiddenActivation = this.hiddenActivation,
          recurrentConnectionType = this.recurrentConnectionType,
          inputType = LayerType.Input.Dense)
      }
    }

    this.layers = initLayers.requireNoNulls()
  }

  /**
   * Initialize the weight of all the [BiRNN] contained in the [layers]
   *
   * @return this DeepBiRNN
   */
  fun initialize(): DeepBiRNN {
    this.layers.forEach { it.initialize() }
    return this
  }
}