/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.biaffine

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayerParameters

/**
 * The parameters of the Biaffine layer.
 *
 * @property inputSize1 the size of the first input
 * @property inputSize2 the size of the second input
 * @property outputSize the size of the output
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
class BiaffineLayerParameters(
  internal val inputSize1: Int,
  internal val inputSize2: Int,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  sparseInput: Boolean = false
) : MergeLayerParameters(
  inputsSize = listOf(inputSize1, inputSize2),
  outputSize = outputSize,
  weightsInitializer = weightsInitializer,
  biasesInitializer = biasesInitializer,
  sparseInput = sparseInput
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The weights connected to the first input array.
   */
  val w1 = ParamsArray(this.outputSize, this.inputSize1)

  /**
   * The parameters connected to the second input array.
   */
  val w2 = ParamsArray(this.outputSize, this.inputSize2)

  /**
   * The bias array.
   */
  val b = ParamsArray(this.outputSize)

  /**
   * The weights connected to each the first and the second input arrays.
   */
  val w: List<ParamsArray> = List(
    size = this.outputSize,
    init = { ParamsArray(this.inputSize2, this.inputSize1) }
  )

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(this.w1, this.w2) + this.w

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
    this.b
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }
}
