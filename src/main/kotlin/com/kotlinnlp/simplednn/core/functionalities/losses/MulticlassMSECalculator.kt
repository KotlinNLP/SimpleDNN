/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 *
 * MeanSquaredErrorCalculator
 */
class MulticlassMSECalculator : MSECalculator() {

  /**
   * Calculate the errors between an output and its gold.
   *
   * @param output the output prediction
   * @param outputGold the expected output
   *
   * @return the derivative of the loss within [output] and [outputGold]
   */
  override fun calculateErrors(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {

    val lossDerivative = output.copy()

    (0 until output.length).forEach { i -> if (outputGold[i] == 1.0) lossDerivative[i] = output[i] - 1.0 }

    return lossDerivative
  }
}
