/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of the layer of type AttentionLayer.
 *
 * @property inputSize the size of each element of the input sequence
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param sparseInput whether the weights connected to the input are sparse or not
 */
class AttentionMechanismLayerParameters(
  inputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  private val sparseInput: Boolean = false
) : LayerParameters(
  inputSize = inputSize,
  outputSize = -1, // depends on the number of element in the input sequence
  weightsInitializer = weightsInitializer,
  biasesInitializer = null
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The context vector trainable parameter.
   */
  val contextVector = ParamsArray(inputSize)

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(this.contextVector)

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf()

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }
}
