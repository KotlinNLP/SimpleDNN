/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils

import com.jsoniter.JsonIterator
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory

/**
 *
 */
fun JsonIterator.readDenseNDArray(): DenseNDArray {

  val array = ArrayList<Double>()

  while (this.readArray()) array.add(this.readDouble())

  return DenseNDArrayFactory.arrayOf(array.toDoubleArray())
}

/**
 *
 */
fun JsonIterator.readSparseBinaryNDArray(): SparseBinaryNDArray {

  val array = ArrayList<Int>()
  var index = 0

  while (this.readArray()) {

    if (Math.round(this.readDouble()).toInt() == 1) {
      array.add(index)
    }

    index++
  }

  return SparseBinaryNDArrayFactory.arrayOf(activeIndices = array.toIntArray(), shape = Shape(784))
}
