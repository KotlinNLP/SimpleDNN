/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning.recirculation

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.deeplearning.newrecirculation.NewRecirculationModel
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The model of [CLRecirculationNetwork].
 *
 * @property classes the set of possible classes
 * @property inputSize the size of the input layer
 * @property hiddenSize the size of the hidden layer
 * @property hiddenActivation the activation function of the hidden layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class CLRecirculationModel(
  val classes: Set<Int>,
  val inputSize: Int,
  val hiddenSize: Int,
  val hiddenActivation: ActivationFunction = Sigmoid(),
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : Serializable {

  /**
   * The map that associates each class to a new-recirculation network.
   */
  val autoencodersModels: Map<Int, NewRecirculationModel> = this.classes.associate {
    it to NewRecirculationModel(
      inputSize = this.inputSize,
      hiddenSize = this.hiddenSize,
      activationFunction = this.hiddenActivation,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer
    )
  }

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [CLRecirculationModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [CLRecirculationModel]
     *
     * @return the [CLRecirculationModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): CLRecirculationModel = Serializer.deserialize(inputStream)
  }

  /**
   * Serialize this [CLRecirculationModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [CLRecirculationModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
