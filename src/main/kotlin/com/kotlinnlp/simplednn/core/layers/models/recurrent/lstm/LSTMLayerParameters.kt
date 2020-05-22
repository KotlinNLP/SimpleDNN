/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLinearParams
import com.kotlinnlp.simplednn.core.layers.LayerParameters

/**
 * The parameters of the layer of type LSTM.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @param sparseInput whether the weights connected to the input are sparse or not
 */
class LSTMLayerParameters(
  inputSize: Int,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  private val sparseInput: Boolean = false
) : LayerParameters(
  inputSize = inputSize,
  outputSize = outputSize,
  weightsInitializer = weightsInitializer,
  biasesInitializer = biasesInitializer
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The input gate parameters.
   */
  val inputGate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   * The output gate parameters.
   */
  val outputGate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   * The forget gate parameters.
   */
  val forgetGate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   * The candidate parameters.
   */
  val candidate = RecurrentLinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(

    this.inputGate.weights,
    this.outputGate.weights,
    this.forgetGate.weights,
    this.candidate.weights,

    this.inputGate.recurrentWeights,
    this.outputGate.recurrentWeights,
    this.forgetGate.recurrentWeights,
    this.candidate.recurrentWeights
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
    this.inputGate.biases,
    this.outputGate.biases,
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
   * Adding a bias of size 1 to the forget gate improves the general performance of the LSTM.
   * (Greff et al., 2015 and Jozefowicz et al., 2015)
   */
  fun initForgetGateBiasToOne() = ConstantInitializer(1.0).initialize(this.forgetGate.biases.values)
}
