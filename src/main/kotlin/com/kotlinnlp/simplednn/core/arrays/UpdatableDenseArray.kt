/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The [UpdatableDenseArray] is a wrapper of a [DenseNDArray]
 */
class UpdatableDenseArray(override val values: DenseNDArray) : UpdatableArray<DenseNDArray>(values = values) {

  /**
   * Build an [UpdatableDenseArray] with values initialized to zeros.
   *
   * @param shape the shape of the [values] array
   *
   * @return a new array with values initialized to zeros
   */
  constructor(shape: Shape) : this(DenseNDArrayFactory.zeros(shape))

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
