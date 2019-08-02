/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.affine

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayerParameters

/**
 * The parameters of the Affine layer.
 *
 * @property inputsSize the size of each input
 * @property outputSize the size of the output
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 * @property sparseInput whether the weights connected to the input are sparse or not
 */
open class AffineLayerParameters(
  inputsSize: List<Int>,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer(),
  sparseInput: Boolean = false
) : MergeLayerParameters(
  inputsSize = inputsSize,
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
   * The weights arrays.
   */
  val w: List<ParamsArray> = inputsSize.map { ParamsArray(dim1 = this.outputSize, dim2 = it) }

  /**
   * The bias array.
   */
  val b = ParamsArray(this.outputSize)

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = this.w

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(this.b)

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }
}
