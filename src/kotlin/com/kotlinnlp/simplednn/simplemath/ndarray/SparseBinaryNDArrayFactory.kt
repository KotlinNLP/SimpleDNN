/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

/**
 *
 */
class SparseBinaryNDArrayFactory : NDArrayFactory<SparseBinaryNDArray> {

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

}
