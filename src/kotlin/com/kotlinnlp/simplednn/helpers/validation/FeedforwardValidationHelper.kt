/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.helpers.validation

import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.dataset.SimpleExample
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.OutputEvaluationFunction

/**
 *
 */
class FeedforwardValidationHelper(override val neuralProcessor: FeedforwardNeuralProcessor,
                                  outputEvaluationFunction: OutputEvaluationFunction): ValidationHelper<SimpleExample>(
  neuralProcessor, outputEvaluationFunction) {

  override fun validate(example: SimpleExample): Boolean {

    val output = this.neuralProcessor.forward(example.features)
    return this.outputEvaluationFunction(output = output, outputGold = example.outputGold)
  }
}
