/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.product

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.mergelayers.MergeLayerParameters

/**
 * The parameters of the Product layer.
 *
 * @property inputSize the size of each input
 */
class ProductLayerParameters(
  inputSize: Int
) : MergeLayerParameters<ProductLayerParameters>(
  inputsSize = listOf(inputSize),
  outputSize = inputSize,
  weightsInitializer = null,
  biasesInitializer = null,
  sparseInput = false
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Check input arrays size.
   */
  init {
    require(this.inputsSize.all { it == this.inputSize })
  }

  /**
   * The list of all parameters.
   */
  override val paramsList = emptyArray<UpdatableArray<*>>()

  /**
   * The list of weights parameters.
   */
  override val weightsList = emptyList<UpdatableArray<*>>()

  /**
   * The list of biases parameters.
   */
  override val biasesList = emptyList<UpdatableArray<*>>()

  /**
   * @return a new [ProductLayerParameters] containing a copy of all parameters of this
   */
  override fun copy() = ProductLayerParameters(inputSize = this.inputSize)
}
