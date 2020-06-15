/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package traininghelpers.validation

import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import utils.SimpleExample
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.OutputEvaluationFunction
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.simplednn.helpers.Evaluator
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A helper which evaluates a dataset of [SimpleExample]s.
 *
 * @param model the model to validate
 * @param examples a list of examples to validate
 * @param outputEvaluationFunction the output evaluation function
 * @param saveContributions whether to save the contributions of each input to its output (needed to calculate the
 *                          relevance, default false)
 * @param afterEachEvaluation a callback called after each example evaluation
 * @param verbose whether to print info about the validation progress (default = true)
 */
class FeedforwardEvaluator<NDArrayType: NDArray<NDArrayType>>(
  model: StackedLayersParameters,
  examples: List<SimpleExample<NDArrayType>>,
  private val outputEvaluationFunction: OutputEvaluationFunction,
  private val saveContributions: Boolean = false,
  private val afterEachEvaluation: (example: SimpleExample<NDArrayType>,
                                    isCorrect: Boolean,
                                    processor: FeedforwardNeuralProcessor<NDArrayType>) -> Unit = { _, _, _ -> },
  verbose: Boolean = true
) : Evaluator<SimpleExample<NDArrayType>, Statistics.Simple>(
  examples = examples,
  verbose = verbose
) {

  /**
   * The validation statistics.
   */
  override val stats: Statistics.Simple = Statistics.Simple()

  /**
   * A feed-forward neural processor.
   */
  private val neuralProcessor = FeedforwardNeuralProcessor<NDArrayType>(model = model, propagateToInput = false)

  /**
   * Evaluate the model with a single example.
   *
   * @param example the example to validate the model with
   */
  override fun evaluate(example: SimpleExample<NDArrayType>) {

    val output: DenseNDArray =
      this.neuralProcessor.forward(example.features, saveContributions = this.saveContributions)

    val isCorrect: Boolean = this.outputEvaluationFunction(output = output, outputGold = example.outputGold)

    if (isCorrect)
      this.stats.metric.truePos++
    else
      this.stats.metric.falsePos++

    this.stats.accuracy = this.stats.metric.precision

    this.afterEachEvaluation(example, isCorrect, this.neuralProcessor)
  }
}
