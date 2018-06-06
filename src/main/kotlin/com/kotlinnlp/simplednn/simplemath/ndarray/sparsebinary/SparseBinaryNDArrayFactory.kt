/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary

import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.VectorIndices

/**
 *
 */
object SparseBinaryNDArrayFactory : NDArrayFactory<SparseBinaryNDArray> {

  /**
   * Private val used to serialize the class (needed from Serializable)
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * @param shape shape
   *
   * @return a new empty [SparseBinaryNDArray]
   */
  override fun emptyArray(shape: Shape): SparseBinaryNDArray {
    TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
  }

  /**
   *
   * @param shape shape
   * @return a new [SparseBinaryNDArray] filled with zeros
   */
  override fun zeros(shape: Shape): SparseBinaryNDArray {
    TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
  }

  /**
   * Build a new [SparseBinaryNDArray] filled with zeros but one with 1.0
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   * @return a oneHotEncoder [SparseBinaryNDArray]
   */
  override fun oneHotEncoder(length: Int, oneAt: Int): SparseBinaryNDArray {
    TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
  }

  /**
   * Build a new [SparseBinaryNDArray] filled with random values uniformly distributed in range [[from], [to]]
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to inclusive upper bound of random values range
   * @return a new [SparseBinaryNDArray] filled with random values
   */
  override fun random(shape: Shape, from: Double, to: Double): SparseBinaryNDArray {
    TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
  }

  /**
   *
   */
  fun arrayOf(activeIndices: List<Int>, shape: Shape): SparseBinaryNDArray {
    require(shape.dim1 == 1 || shape.dim2 == 1) {
      "Invalid shape (only a 1-dim SparseBinaryNDArray can be created given a list of active indices)"
    }

    val vectorMap = mutableMapOf<Int, VectorIndices?>(Pair(0, activeIndices.toMutableList()))
    val indicesMap = mutableMapOf<Int, VectorIndices?>()

    for (index in activeIndices) {
      indicesMap[index] = null
    }

    return if (shape.dim1 == 1)
      SparseBinaryNDArray(activeIndicesByRow = vectorMap, activeIndicesByColumn = indicesMap, shape = shape)
    else
      SparseBinaryNDArray(activeIndicesByRow = indicesMap, activeIndicesByColumn = vectorMap, shape = shape)
  }

  /**
   *
   */
  fun arrayOf(activeIndices: Array<Indices>, shape: Shape): SparseBinaryNDArray {

    val res = SparseBinaryNDArray(shape = shape)

    for ((i, j) in activeIndices) {
      res.set(i, j)
    }

    return res
  }
}
