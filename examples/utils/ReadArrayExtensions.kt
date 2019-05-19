/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils

import com.beust.klaxon.JsonArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory

/**
 *
 */
fun JsonArray<*>.readDenseNDArray(): DenseNDArray =
  DenseNDArrayFactory.arrayOf(DoubleArray(size = this.size, init = { i -> (this[i] as Number).toDouble() }))

/**
 *
 */
fun JsonArray<*>.readSparseBinaryNDArray(size: Int): SparseBinaryNDArray =
  SparseBinaryNDArrayFactory.arrayOf(activeIndices = this.map { it as Int }.sorted(), shape = Shape(size))

/**
 *
 */
fun JsonArray<*>.readSparseBinaryNDArrayFromDense(size: Int): SparseBinaryNDArray {

  val activeIndices: List<Int> =
    this.mapIndexed { index, it -> if (it as Double > 0.5) index else -1 }.filter { it > 0 }

  return SparseBinaryNDArrayFactory.arrayOf(activeIndices = activeIndices.sorted(), shape = Shape(size))
}
