/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.autoassociative

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.models.LinearParams

/**
 * The parameters of the layer of type NewRecirculation.
 *
 * @property inputSize input size (the output size will be the same)
 * @property hiddenSize hiddenSize the size of the hidden layer
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class NewRecirculationLayerParameters(
  inputSize: Int,
  val hiddenSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : LayerParameters<NewRecirculationLayerParameters>(
  inputSize = inputSize,
  outputSize = inputSize,
  weightsInitializer = weightsInitializer,
  biasesInitializer = biasesInitializer) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   */
  val unit = LinearParams(
    inputSize = this.inputSize,
    outputSize = this.hiddenSize,
    sparseInput = false)

  /**
   * The list of all parameters.
   */
  override val paramsList = listOf(
    this.unit.weights,
    this.unit.biases
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(
    this.unit.weights
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
    this.unit.biases
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [NewRecirculationLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): NewRecirculationLayerParameters {

    val clonedParams = NewRecirculationLayerParameters(
      inputSize = this.inputSize,
      hiddenSize = this.hiddenSize,
      weightsInitializer = null,
      biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
