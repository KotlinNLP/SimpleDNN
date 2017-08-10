/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.arrays

import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The [DistributionArray] is a wrapper of an [NDArray] in which values represent a probability distribution.
 *
 * @property values the values as [DenseNDArray]
 */
class DistributionArray(values: DenseNDArray) : Norm1Array<DenseNDArray>(values) {

  companion object {

    /**
     * A factory of [DistributionArray]s representing a uniform distribution.
     *
     * @param length the length of the array
     *
     * @return a [DistributionArray] with the values initialized to a uniform distribution
     */
    fun uniform(length: Int): DistributionArray {

      val initValue = 1.0 / length
      val values = DoubleArray(size = length, init = { initValue })

      return DistributionArray(values = DenseNDArrayFactory.arrayOf(values))
    }

    /**
     * A factory of [DistributionArray]s representing a distribution concentrated on one element,
     * with all the others equal to 0.0.
     *
     * @param length the length of the array
     * @param oneAt the index of the 1.0 element
     *
     * @return a [DistributionArray] with the values initialized to a uniform distribution
     */
    fun oneHot(length: Int, oneAt: Int): DistributionArray {
      require(oneAt < length) { "The index of the 1.0 element exceeds the length of the array" }

      val values = DenseNDArrayFactory.oneHotEncoder(length = length, oneAt = oneAt)

      return DistributionArray(values)
    }
  }

  init {
    require((0 until this.values.length).all{ i -> this.values[i] in 0.0 .. 1.0}) { "Required 0 <= value[i] <= 1.0" }
    require(equals(this.values.sum(), 1.0, tolerance = 1.0e-08)) { "Values sum must be equal to 1.0" }
    require(this.values.columns == 1) { "Values must be a column vector" }
  }

  /**
   * Assign values to the array.
   *
   * @param values values to assign to this [DistributionArray]
   */
  override fun assignValues(values: NDArray<*>) {
    require((0 until this.values.length).all{ i -> this.values[i] in 0.0 .. 1.0}) { "Required 0 <= value[i] <= 1.0" }
    require(equals(values.sum(), 1.0, tolerance = 1.0e-08)) { "Values sum must be equal to 1.0" }

    this.values.assignValues(values)
  }

  /**
   * Clone this array.
   *
   * @return a clone of this [DistributionArray]
   */
  override fun clone(): DistributionArray = DistributionArray(values = this.values.copy())
}
