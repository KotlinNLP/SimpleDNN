/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.training

import com.kotlinnlp.simplednn.core.functionalities.losses.LossCalculator
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.dataset.SimpleExample

/**
 *
 */
class FeedforwardTrainingHelper(
  override val neuralProcessor: FeedforwardNeuralProcessor,
  optimizer: ParamsOptimizer,
  lossCalculator: LossCalculator,
  verbose: Boolean = false
) : TrainingHelper<SimpleExample>(
  neuralProcessor = neuralProcessor,
  optimizer = optimizer,
  lossCalculator = lossCalculator,
  verbose = verbose) {

  /**
   * Learn from an example (forward + backward)
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output respect to the gold
   */
  override fun learnFromExample(example: SimpleExample): Double {

    val output = this.neuralProcessor.forward(example.features)
    val errors = this.lossCalculator.calculateErrors(output, example.outputGold)

    this.neuralProcessor.backward(errors)

    return this.lossCalculator.calculateLoss(output, example.outputGold).avg()
  }
}
