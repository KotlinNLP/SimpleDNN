/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.*

/**
 * The [Norm1Array] is a wrapper of an [NDArray] in which values represent a vector with norm equals to 1.
 *
 * @property values the values as [NDArray]
 */
open class Norm1Array<NDArrayType : NDArray<NDArrayType>>(val values: NDArrayType) {

  init {
    require(equals(this.values.sum(), 1.0, tolerance = 1.0e-08)) { "Values sum must be equal to 1.0" }
    require(this.values.columns == 1) { "Values must be a column vector" }
  }

  /**
   * The length of this array.
   */
  val length: Int = this.values.length

  /**
   * Assign values to the array.
   *
   * @param values values to assign to this [Norm1Array]
   */
  open fun assignValues(values: NDArray<*>) {
    require(equals(values.sum(), 1.0, tolerance = 1.0e-08)) { "Values sum must be equal to 1.0" }

    this.values.assignValues(values)
  }

  /**
   * Clone this array.
   *
   * @return a clone of this [Norm1Array]
   */
  open fun clone(): Norm1Array<NDArrayType> = Norm1Array(values = this.values.copy())
}
