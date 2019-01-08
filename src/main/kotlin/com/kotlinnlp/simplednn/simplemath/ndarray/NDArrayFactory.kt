/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import java.io.Serializable

/**
 *
 */
interface NDArrayFactory<NDArrayType : NDArray<NDArrayType>> : Serializable {

  /**
   * @param shape shape
   *
   * @return a new empty [NDArrayType]
   */
  fun emptyArray(shape: Shape): NDArrayType

  /**
   * Build a new [NDArrayType] filled with zeros
   *
   * @param shape shape
   *
   * @return a new [NDArrayType]
   */
  fun zeros(shape: Shape): NDArrayType

  /**
   * Build a new [NDArrayType] filled with a given value.
   *
   * @param shape shape
   * @param value the init value
   *
   * @return a new [NDArrayType]
   */
  fun fill(shape: Shape, value: Double): NDArrayType

  /**
   * Build a new [NDArrayType] filled with zeros but one with 1.0
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   *
   * @return a oneHotEncoder [NDArrayType]
   */
  fun oneHotEncoder(length: Int, oneAt: Int): NDArrayType

  /**
   * Build a new [NDArrayType] filled with random values uniformly distributed in range [[from], [to]]
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to inclusive upper bound of random values range
   *
   * @return a new [NDArrayType] filled with random values
   */
  fun random(shape: Shape, from: Double = 0.0, to: Double = 1.0): NDArrayType
}
