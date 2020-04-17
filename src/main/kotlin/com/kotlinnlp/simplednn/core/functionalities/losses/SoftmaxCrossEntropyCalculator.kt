/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import kotlin.math.ln

/**
 * Softmax cross-entropy Calculator.
 *
 * It must be used to calculate the loss ONLY if the activation function of the
 * output is the [com.kotlinnlp.simplednn.core.functionalities.activations.Softmax].
 */
open class SoftmaxCrossEntropyCalculator : LossCalculator {

  companion object {

    /**
     * A values threshold to avoid underflow errors.
     */
    private const val EPS: Double = 1.0e-08
  }

  /**
   * Calculate the loss between an output and its gold.
   *
   *   -G * log(O)
   *
   * @param output the output prediction
   * @param outputGold the expected output
   *
   * @return the loss within [output] and [outputGold]
   */
  override fun calculateLoss(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {

    require(outputGold.isOneHotEncoder) {
      "The gold output must be a one hot encoder to calculate the loss with the cross-entropy function."
    }

    val goldIndex: Int = outputGold.argMaxIndex()
    val loss: DenseNDArray = output.zerosLike()

    loss[goldIndex] = -ln(maxOf(EPS, output[goldIndex]))

    return loss
  }

  /**
   * Calculate the errors between an output and its gold.
   *
   * @param output the output prediction
   * @param outputGold the expected output
   *
   * @return the derivative of the loss within [output] and [outputGold]
   */
  override fun calculateErrors(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray =
    output.sub(outputGold)

  /**
   * Calculate the errors between an output and its gold, given the one hot gold index.
   *
   * @param output the output prediction
   * @param goldIndex the index of the gold value
   *
   * @return the derivative of the loss within the [output] and its gold
   */
  fun calculateErrors(output: DenseNDArray, goldIndex: Int): DenseNDArray {

    val errors: DenseNDArray = output.copy()

    errors[goldIndex] = errors[goldIndex] - 1.0

    return errors
  }

  /**
   * Calculate the errors of a sequence.
   *
   * TODO: the [outputGoldSequence] should be a List<Int>
   *
   * @param outputSequence a list containing the output of the network for each example of a sequence
   * @param outputGoldSequence a list containing the indexes of the gold sequence
   *
   * @return an array containing the errors for each example of the sequence
   */
  fun calculateErrors(outputSequence: List<DenseNDArray>,
                      outputGoldSequence: ArrayList<Int>): List<DenseNDArray> {

    require(outputSequence.size == outputGoldSequence.size)

    return outputSequence.zip(outputGoldSequence).map {
      this.calculateErrors(output = it.first, goldIndex = it.second)
    }
  }
}
