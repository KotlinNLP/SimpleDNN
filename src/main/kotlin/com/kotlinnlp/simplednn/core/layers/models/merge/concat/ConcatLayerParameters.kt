/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concat

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayerParameters

/**
 * The parameters of the Concat layer.
 *
 * @property inputsSize the size of each input
 */
class ConcatLayerParameters(
  inputsSize: List<Int>
) : MergeLayerParameters<ConcatLayerParameters>(
  inputsSize = inputsSize,
  outputSize = inputsSize.sum(),
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
   * The list of all parameters.
   */
  override val paramsList = emptyList<ParamsArray>()

  /**
   * The list of weights parameters.
   */
  override val weightsList = emptyList<ParamsArray>()

  /**
   * The list of biases parameters.
   */
  override val biasesList = emptyList<ParamsArray>()

  /**
   * @return a new [ConcatLayerParameters] containing a copy of all parameters of this
   */
  override fun copy() = ConcatLayerParameters(inputsSize = this.inputsSize)
}
