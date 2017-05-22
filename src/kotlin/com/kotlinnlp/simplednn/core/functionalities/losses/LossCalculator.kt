/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.NDArray

/**
 *
 */
interface LossCalculator {

  /**
   * Calculate the loss within an output and its gold.
   *
   * @param output  current output layer
   * @param outputGold expected binary output
   *
   * @return the loss within [output] and [outputGold]
   */
  fun calculateLoss(output: NDArray, outputGold: NDArray): NDArray

  /**
   * Calculate the errors within an output and its gold.
   *
   * @param output current output layer
   * @param outputGold expected binary output
   *
   * @return the derivative of the loss within [output] and [outputGold]
   */
  fun calculateErrors(output: NDArray, outputGold: NDArray): NDArray

  /**
   * Calculate the errors of a sequence.
   *
   * @param outputSequence an Array containing the output of the network for each example of a sequence
   * @param outputGoldSequence an ArrayList containing the output gold sequence
   *
   * @return an array containing the errors for each example of the sequence
   */
  fun calculateErrors(outputSequence: Array<NDArray>,
                      outputGoldSequence: Array<NDArray>): Array<NDArray> {

    require(outputSequence.size == outputGoldSequence.size)

    return Array(
      size = outputSequence.size,
      init = { this.calculateErrors(outputSequence[it], outputGoldSequence[it]) })
  }

  /**
   * Calculate the mean loss within a sequence.
   *
   * @param outputSequence an Array containing the output of the network for each example of a sequence
   * @param outputGoldSequence an ArrayList containing the output gold sequence
   *
   * @return the mean loss of the sequence
   */
  fun calculateMeanLoss(outputSequence: Array<NDArray>,
                        outputGoldSequence: Array<NDArray>): Double {

    var lossesSum: Double = 0.0

    outputSequence.zip(outputGoldSequence).forEach { (output, goldOutput) ->
      lossesSum += this.calculateLoss(output, goldOutput).avg()
    }

    return lossesSum / outputGoldSequence.size
  }
}
