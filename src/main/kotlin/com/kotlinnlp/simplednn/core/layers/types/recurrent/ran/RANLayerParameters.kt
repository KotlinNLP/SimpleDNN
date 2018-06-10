/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.recurrent.ran

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.types.recurrent.RecurrentParametersUnit
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.types.ParametersUnit

/**
 * The parameters of the layer of type RAN.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @param sparseInput whether the weights connected to the input are sparse or not
 * @param meProp whether to use the 'meProp' errors propagation algorithm (params are sparse)
 */
class RANLayerParameters(
  inputSize: Int,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  private val sparseInput: Boolean = false,
  private val meProp: Boolean = false
) : LayerParameters<RANLayerParameters>(
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
  val inputGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput,
    meProp = this.meProp)

  /**
   *
   */
  val forgetGate = RecurrentParametersUnit(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput,
    meProp = this.meProp)

  /**
   *
   */
  val candidate = ParametersUnit(
    inputSize = inputSize,
    outputSize = outputSize,
    sparseInput = this.sparseInput,
    meProp = this.meProp)

  /**
   * The list of all parameters.
   */
  override val paramsList = listOf(

    this.inputGate.weights,
    this.forgetGate.weights,
    this.candidate.weights,

    this.inputGate.biases,
    this.forgetGate.biases,
    this.candidate.biases,

    this.inputGate.recurrentWeights,
    this.forgetGate.recurrentWeights
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<UpdatableArray<*>> = listOf(

    this.inputGate.weights,
    this.forgetGate.weights,
    this.candidate.weights,

    this.inputGate.recurrentWeights,
    this.forgetGate.recurrentWeights
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<UpdatableArray<*>> = listOf(
    this.inputGate.biases,
    this.forgetGate.biases,
    this.candidate.biases
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }

  /**
   * @return a new [RANLayerParameters] containing a copy of all parameters of this
   */
  override fun copy(): RANLayerParameters {

    val clonedParams = RANLayerParameters(
      inputSize = this.inputSize,
      outputSize = this.outputSize,
      sparseInput = this.sparseInput,
      weightsInitializer = null,
      biasesInitializer = null)

    clonedParams.assignValues(this)

    return clonedParams
  }
}
