/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training

import com.kotlinnlp.simplednn.dataset.BinaryOutputExample
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward.CLFeedforwardNetwork
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward.CLFeedforwardNetworkOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Training Helper for the Competitive Learning FeeNetwork.
 *
 * @property optimizer the optimizer
 * @property verbose whether to print training details
 */
class CLFeedforwardTrainingHelper(
  val network: CLFeedforwardNetwork,
  override val optimizer: CLFeedforwardNetworkOptimizer,
  verbose: Boolean = false
) : TrainingHelper<BinaryOutputExample<DenseNDArray>>(optimizer = optimizer, verbose = verbose) {

  /**
   * The class index of the last example learned.
   */
  private var lastExampleClass: Int = -1

  /**
   * Learn from an example (forward + backward).
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output respect to the gold
   */
  override fun learnFromExample(example: BinaryOutputExample<DenseNDArray>): Double {

    this.lastExampleClass = example.goldOutcomeIndex

    return this.network.learn(inputArray = example.features, classIndex = example.goldOutcomeIndex)
  }

  /**
   * Accumulate the params errors resulting from [learnFromExample].
   *
   * @param batchSize the size of each batch
   */
  override fun accumulateParamsErrors(batchSize: Int) {

    this.optimizer.accumulate(
      paramsErrors = Pair(this.lastExampleClass, this.network.getParamsErrors(copy = batchSize > 1)),
      copy = batchSize > 1)
  }
}
