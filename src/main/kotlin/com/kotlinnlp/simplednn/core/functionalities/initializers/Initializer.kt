/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.initializers

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.io.Serializable

/**
 * An initializer of the values of dense arrays.
 */
interface Initializer : Serializable {

  /**
   * Initialize the values of the given [array].
   *
   * @param array a dense array
   */
  fun initialize(array: DenseNDArray)
}
