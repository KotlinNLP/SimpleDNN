/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The [UpdatableSparseArray] is a wrapper of a [SparseNDArray]
 */
class UpdatableSparseArray(override val values: SparseNDArray) : UpdatableArray<SparseNDArray>(values = values) {

  /**
   * Build an [UpdatableSparseArray] with values initialized to zeros.
   *
   * @param shape the shape of the [values] array
   *
   * @return a new array with values initialized to zeros
   */
  constructor(shape: Shape) : this(SparseNDArrayFactory.zeros(shape))

  /**
   * Build an [UpdatableSparseArray] with values initialized to zeros.
   *
   * @param dim1 the first dimension of the array
   * @param dim2 the second dimension of the array (default = 1)
   *
   * @return a new dense array with values initialized to zeros
   */
  constructor(dim1: Int, dim2: Int = 1) : this(Shape(dim1, dim2))

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
