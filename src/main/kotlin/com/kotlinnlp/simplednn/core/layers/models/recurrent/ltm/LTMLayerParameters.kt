/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.models.LinearParams
import com.kotlinnlp.simplednn.core.optimizer.ParamsList

/**
 * The parameters of the layer of type LTM.
 *
 * @property inputSize input size
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param sparseInput whether the weights connected to the input are sparse or not
 */
class LTMLayerParameters(
  inputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  private val sparseInput: Boolean = false
) : LayerParameters(
  inputSize = inputSize,
  outputSize = inputSize,
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
   * The parameters of the L1 input gate.
   */
  val inputGate1: LinearParams = this.buildGateParams()

  /**
   * The parameters of the L2 input gate.
   */
  val inputGate2: LinearParams = this.buildGateParams()

  /**
   * The parameters of the L3 input gate.
   */
  val inputGate3: LinearParams = this.buildGateParams()

  /**
   * The parameters of the cell.
   */
  val cell: LinearParams = this.buildGateParams()

  /**
   * The list of weights parameters.
   */
  override val weightsList: ParamsList

  /**
   * The list of biases parameters.
   */
  override val biasesList: ParamsList = listOf()

  /**
   * Initialize all parameters values.
   */
  init {

    val params: List<LinearParams> = listOf(this.inputGate1, this.inputGate2, this.inputGate3, this.cell)

    this.weightsList = params.map { it.weights }

    this.initialize()
  }

  /**
   * @return the parameters of a gate
   */
  private fun buildGateParams() = LinearParams(
    inputSize = this.inputSize,
    outputSize = this.outputSize,
    sparseInput = this.sparseInput)
}
