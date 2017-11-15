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
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

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
                val numberOfLayers: Int = 1) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [DeepBiRNN] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [DeepBiRNN]
     *
     * @return the [DeepBiRNN] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): BiRNN = Serializer.deserialize(inputStream)
  }

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

  /**
   * Serialize this [DeepBiRNN] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [DeepBiRNN]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}