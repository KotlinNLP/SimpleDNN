/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.regularization

import com.kotlinnlp.simplednn.core.arrays.ParamsArray

/**
 * Regularization method based on the Euclidean norm.
 */
object MaxNormRegularization : ParamsRegularization {

  /**
   * Apply the regularization to given parameters.
   *
   * @param params the parameters to regularize
   */
  override fun apply(params: ParamsArray) {

    val norm2 = params.values.norm2()

    (0 until params.values.length)
      .filter { params.values[it] > norm2 }
      .forEach { params.values[it] = norm2 }
  }
}
