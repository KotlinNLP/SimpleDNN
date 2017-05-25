/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.ndarray.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseBinaryNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The [UpdatableSparseBinaryArray] is a wrapper of a [SparseBinaryNDArray]
 */
class UpdatableSparseBinaryArray(
  override val values: SparseBinaryNDArray
) : UpdatableArray<SparseBinaryNDArray>(values = values) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     *
     */
    operator fun invoke(shape: Shape) = UpdatableSparseBinaryArray(SparseBinaryNDArrayFactory.zeros(shape))
  }
}
