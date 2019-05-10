/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.models.LinearParams

/**
 * The parameters associated to a [RecurrentLayerUnit].
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param sparseInput whether the weights connected to the input are sparse or not (default false)
 */
class RecurrentLinearParams(
  inputSize: Int,
  outputSize: Int,
  sparseInput: Boolean = false
) : LinearParams(
  inputSize = inputSize,
  outputSize = outputSize,
  sparseInput = sparseInput) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The recurrent weights initialized to zero.
   */
  val recurrentWeights = ParamsArray(dim1 = this.outputSize, dim2 = this.outputSize)
}
