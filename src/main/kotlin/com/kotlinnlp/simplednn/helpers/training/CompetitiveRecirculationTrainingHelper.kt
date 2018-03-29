/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training

import com.kotlinnlp.simplednn.dataset.BinaryOutputExample
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.recirculation.CLRecirculationNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Training Helper for the Competitive Learning based on the New Recirculation.
 *
 * @property optimizer the optimizer
 * @property verbose whether to print training details
 */
class CompetitiveRecirculationTrainingHelper(
  val network: CLRecirculationNetwork,
  verbose: Boolean = false
) : TrainingHelper<BinaryOutputExample<DenseNDArray>>(
  optimizer = null,
  verbose = verbose) {

  /**
   * Learn from an example (forward + backward)
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output respect to the gold
   */
  override fun learnFromExample(example: BinaryOutputExample<DenseNDArray>): Double =
    this.network.learn(inputArray = example.features, classId = example.goldOutcomeIndex)

  /**
   * Accumulate the params errors resulting from [learnFromExample].
   *
   * @param batchSize the size of each batch
   */
  override fun accumulateParamsErrors(batchSize: Int) = Unit
}
