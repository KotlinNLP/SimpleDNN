/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.newrecirculation

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.types.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of the [NewRecirculationNetwork].
 *
 * @property inputSize the size of the input layer
 * @property hiddenSize the size of the hidden layer
 * @property activationFunction the activation function of the output (can be null, default: Sigmoid)
 * @property lambda the partition factor (default = 0.75)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class NewRecirculationModel(
  val inputSize: Int,
  val hiddenSize: Int,
  val activationFunction: ActivationFunction? = Sigmoid(),
  val lambda: Double = 0.75,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [NewRecirculationModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [NewRecirculationModel]
     *
     * @return the [NewRecirculationModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): NewRecirculationModel = Serializer.deserialize(inputStream)
  }

  /**
   * The network parameters.
   */
  val params = FeedforwardLayerParameters(
    inputSize = this.inputSize,
    outputSize = this.hiddenSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer
  )

  /**
   * Serialize this [NewRecirculationModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [NewRecirculationModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
