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
interface NDArrayFactory {

  /**
   *
   * @param shape shape
   * @return a new empty NDArray
   */
  fun emptyArray(shape: Shape): NDArrayInterface

  /**
   *
   */
  fun arrayOf(vector: DoubleArray): NDArrayInterface

  /**
   *
   */
  fun arrayOf(matrix: Array<DoubleArray>): NDArrayInterface

  /**
   *
   * @param shape shape
   * @return a new NDArray filled with zeros
   */
  fun zeros(shape: Shape): NDArrayInterface

  /**
   * Build a new NDArrayInterface filled with zeros but one with 1.0
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   * @return a oneHotEncoder NDArrayInterface
   */
  fun oneHotEncoder(length: Int, oneAt: Int): NDArrayInterface

  /**
   * Build a new NDArrayInterface filled with random values uniformly distributed in range [[from], [to])
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to exclusive upper bound of random values range
   * @return a new NDArray filled with random values
   */
  fun random(shape: Shape, from: Double = 0.0, to: Double = 1.0): NDArrayInterface
}
