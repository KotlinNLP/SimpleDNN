/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import java.io.Serializable

/**
 * The parameters for a linear transformation.
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param sparseInput whether the weights connected to the input are sparse or not (default false)
 */
open class LinearParams(
  val inputSize: Int,
  val outputSize: Int,
  private val sparseInput: Boolean = false
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   */
  val biases = ParamsArray(this.outputSize)

  /**
   *
   */
  val weights = ParamsArray(
    dim1 = this.outputSize,
    dim2 = this.inputSize
  )
}
