/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory

/**
 *
 */
object UpdateMethodsUtils {

  /**
   *
   */
  fun buildParamsArray(): ParamsArray {

    val values: DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.4, 0.5, 1.0, 0.8))
    val array = ParamsArray(DenseNDArrayFactory.zeros(values.shape))

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
  fun buildDenseErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.7, 0.4, 0.8, 0.1))

  /**
   *
   */
  fun buildSparseErrors() = SparseNDArrayFactory.arrayOf(
    activeIndicesValues = arrayOf(
      SparseEntry(Indices(1, 0), 0.7),
      SparseEntry(Indices(4, 0), 0.3)
    ),
    shape = Shape(5)
  )
}
