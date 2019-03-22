/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.gru

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLinearParams
import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of the layer of type GRU.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @param sparseInput whether the weights connected to the input are sparse or not
 */
class GRULayerParameters(
  inputSize: Int,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  private val sparseInput: Boolean = false
) : LayerParameters<GRULayerParameters>(
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
   *
   */
  val candidate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val resetGate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   *
   */
  val partitionGate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   * The list of all parameters.
   */
  override val paramsList = listOf(

    this.candidate.weights,
    this.resetGate.weights,
    this.partitionGate.weights,

    this.candidate.biases,
    this.resetGate.biases,
    this.partitionGate.biases,

    this.candidate.recurrentWeights,
    this.resetGate.recurrentWeights,
    this.partitionGate.recurrentWeights
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(

    this.candidate.weights,
    this.resetGate.weights,
    this.partitionGate.weights,

    this.candidate.recurrentWeights,
    this.resetGate.recurrentWeights,
    this.partitionGate.recurrentWeights
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
    this.candidate.biases,
    this.resetGate.biases,
    this.partitionGate.biases
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [GRULayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): GRULayerParameters {

    val clonedParams = GRULayerParameters(
      inputSize = this.inputSize,
      outputSize = this.outputSize,
      sparseInput = this.sparseInput,
      weightsInitializer = null,
      biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
