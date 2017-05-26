/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.regularization

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 *
 */
interface WeightsRegularization {

  /**
   * Regularization parameter
   */
  val lambda: Double

  /**
   * Regularize [weights] before the update
   *
   * @param weights the weights to update
   */
  fun <NDArrayType: NDArray<NDArrayType>> apply(weights: UpdatableArray<NDArrayType>)
}
