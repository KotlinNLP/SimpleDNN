/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package updatemethods

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.DenseNDArrayFactory

/**
 *
 */
object Utils {

  /**
   *
   */
  fun buildUpdateableArray(): UpdatableDenseArray {
    val values: DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.4, 0.5, 1.0, 0.8))
    val array: UpdatableDenseArray = UpdatableDenseArray(DenseNDArrayFactory.zeros(values.shape))

    array.values.assignValues(values)

    return array
  }

  /**
   *
   */
  fun supportArray1() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, 0.8, 0.5, 0.3, 0.2))

  /**
   *
   */
  fun supportArray2() = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.4, 0.7, 0.0, 0.2))

  /**
   *
   */
  fun buildErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.7, 0.4, 0.8, 0.1))
}
