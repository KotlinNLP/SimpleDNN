/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableSparseArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The parameters associated to a [LayerUnit].
 */
open class ParametersUnit(val inputSize: Int, val outputSize: Int, private val sparseInput: Boolean = false) {

  /**
   *
   */
  val biases: UpdatableDenseArray = this.buildDenseArray(this.outputSize)

  /**
   *
   */
  val weights: UpdatableArray<*> = this.buildUpdatableArray(
    dim1 = this.outputSize, dim2 = this.inputSize, sparseInput = this.sparseInput)

  /**
   *
   */
  private fun buildUpdatableArray(dim1: Int, dim2: Int = 1, sparseInput: Boolean = false): UpdatableArray<*> =
    if (sparseInput)
      this.buildSparseArray(dim1, dim2)
    else
      this.buildDenseArray(dim1, dim2)

  /**
   *
   */
  protected fun buildDenseArray(dim1: Int, dim2: Int = 1) = UpdatableDenseArray(Shape(dim1, dim2))

  /**
   *
   */
  private fun buildSparseArray(dim1: Int, dim2: Int = 1) = UpdatableSparseArray(Shape(dim1, dim2))
}
