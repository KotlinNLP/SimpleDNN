/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Softmax cross-entropy Calculator.
 *
 * It must be used to calculate the loss ONLY if the activation function of the
 * output is the [com.kotlinnlp.simplednn.core.functionalities.activations.Softmax].
 */
open class SoftmaxCrossEntropyCalculator : LossCalculator {

  private val eps: Double = 1.0e-08

  /**
   * Calculate the loss between an output and its gold.
   *
   *   -G * log(O)
   *
   * @param output  current output layer
   * @param outputGold expected binary output
   *
   * @return the loss within [output] and [outputGold]
   */
  override fun calculateLoss(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {
    require(outputGold.isOneHotEncoder) {
      "The gold output must be a one hot encoder to calculate the loss with the cross-entropy function."
    }

    val oneIndex: Int = outputGold.argMaxIndex()
    val loss: DenseNDArray = output.zerosLike()
    val argmaxOutput: Double = output[oneIndex]

    loss[oneIndex] = -Math.log(if (argmaxOutput >= this.eps) argmaxOutput else this.eps)

    return loss
  }

  /**
   * Calculate the errors between an output and its gold.
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
