/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableSparseArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import java.io.Serializable

/**
 * The parameters associated to a [LayerUnit].
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param sparseInput whether the weights connected to the input are sparse or not (default false)
 * @param meProp whether to use the 'meProp' errors propagation algorithm (params are sparse) (default false)
 */
open class ParametersUnit(
  val inputSize: Int,
  val outputSize: Int,
  private val sparseInput: Boolean = false,
  private val meProp: Boolean = false
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   */
  val biases: UpdatableArray<*> = this.buildUpdatableArray(dim1 = this.outputSize, sparse = this.meProp)

  /**
   *
   */
  val weights: UpdatableArray<*> = this.buildUpdatableArray(
    dim1 = this.outputSize, dim2 = this.inputSize, sparse = this.sparseInput || this.meProp)

  /**
   *
   */
  protected fun buildUpdatableArray(dim1: Int, dim2: Int = 1, sparse: Boolean = false): UpdatableArray<*> =
    if (sparse)
      this.buildSparseArray(dim1, dim2)
    else
      this.buildDenseArray(dim1, dim2)

  /**
   * Build an [UpdatableDenseArray] with values initialized to zeros.
   *
   * @param dim1 the first dimension of the array
   * @param dim2 the second dimension of the array (default = 1)
   *
   * @return a new dense array with values initialized to zeros
   */
  private fun buildDenseArray(dim1: Int, dim2: Int = 1) = UpdatableDenseArray(Shape(dim1, dim2))

  /**
   * Build an [UpdatableSparseArray] with values initialized to zeros.
   *
   * @param dim1 the first dimension of the array
   * @param dim2 the second dimension of the array (default = 1)
   *
   * @return a new dense array with values initialized to zeros
   */
  private fun buildSparseArray(dim1: Int, dim2: Int = 1) = UpdatableSparseArray(Shape(dim1, dim2))
}
