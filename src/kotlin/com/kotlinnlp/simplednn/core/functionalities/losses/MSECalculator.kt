/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * MeanSquaredErrorCalculator
 */
open class MSECalculator : LossCalculator {

  /**
   * Calculate the loss within an output and its gold.
   *
   * @param output  current output layer
   * @param outputGold expected binary output
   *
   * @return the loss within [output] and [outputGold]
   */
  override fun calculateLoss(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {
    return output.sub(outputGold).pow(2.0).prod(0.5)
  }

  /**
   * Calculate the errors within an output and its gold.
   *
   * @param output current output layer
   * @param outputGold expected binary output
   *
   * @return the derivative of the loss within [output] and [outputGold]
   */
  override fun calculateErrors(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {
    return output.sub(outputGold)
  }

}
