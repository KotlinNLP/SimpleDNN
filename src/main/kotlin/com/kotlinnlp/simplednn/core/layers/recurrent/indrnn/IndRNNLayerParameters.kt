/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.ParametersUnit

/**
 * The parameters of the layer of type IndRNN.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @param sparseInput whether the weights connected to the input are sparse or not
 */
class IndRNNLayerParameters(
  inputSize: Int,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  private val sparseInput: Boolean = false
) : LayerParameters<IndRNNLayerParameters>(
  inputSize = inputSize,
  outputSize = outputSize,
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
   * The forward parameters.
   */
  val feedforwardUnit = ParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   * The recurrent weights.
   */
  val recurrentWeights: UpdatableArray<*> = this.buildUpdatableArray(dim1 = this.outputSize, sparseInput = false)

  /**
   * The list of all parameters.
   */
  override val paramsList = listOf(
    this.feedforwardUnit.weights,
    this.feedforwardUnit.biases,
    this.recurrentWeights
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<UpdatableArray<*>> = listOf(
    this.feedforwardUnit.weights,
    this.recurrentWeights
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<UpdatableArray<*>> = listOf(
    this.feedforwardUnit.biases
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [IndRNNLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): IndRNNLayerParameters {

    val clonedParams = IndRNNLayerParameters(
      inputSize = this.inputSize,
      outputSize = this.outputSize,
      sparseInput = this.sparseInput,
      weightsInitializer = null,
      biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
