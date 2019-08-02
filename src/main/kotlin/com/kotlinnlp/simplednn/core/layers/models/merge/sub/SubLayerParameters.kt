/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.sub

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayerParameters

/**
 * The parameters of the Distance layer.
 *
 * @property inputSize the size of each input
 */
class SubLayerParameters(
  inputSize: Int
) : MergeLayerParameters(
  inputsSize = List(size = 2, init = { inputSize }),
  outputSize = inputSize,
  weightsInitializer = null,
  biasesInitializer = null,
  sparseInput = false // actually not used because there are no parameters
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
   * The list of weights parameters.
   */
  override val weightsList = emptyList<ParamsArray>()

  /**
   * The list of biases parameters.
   */
  override val biasesList = emptyList<ParamsArray>()
}
