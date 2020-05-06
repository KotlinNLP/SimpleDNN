/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import traininghelpers.training.SequenceTrainer
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.CFN
import com.kotlinnlp.simplednn.helpers.Statistics
import traininghelpers.validation.SequenceEvaluator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.Corpus
import utils.SequenceExample
import utils.CorpusReader
import utils.exampleextractor.ClassificationSequenceExampleExtractor

fun main() {

  println("Start 'Progressive Sum Test'")

  val dataset = CorpusReader<SequenceExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().progressive_sum.datasets_paths, // same for validation and test
    exampleExtractor = ClassificationSequenceExampleExtractor(outputSize = 11),
    perLine = true)

  ProgressiveSumTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
class ProgressiveSumTest(val dataset: Corpus<SequenceExample<DenseNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = CFN(
      inputSize = 1,
      hiddenSize = 100,
      hiddenActivation = Tanh,
      outputSize = 11,
      outputActivation = Softmax())

  /**
   *
   */
  fun start() {

    this.initialValidation()
    this.train()
  }

  /**
   *
   */
  private fun initialValidation() {

    println("\n-- VALIDATION BEFORE TRAINING\n")

    val evaluator = SequenceEvaluator(
        model = this.neuralNetwork,
        examples = this.dataset.validation,
        outputEvaluationFunction = ClassificationEvaluation())

    val stats: Statistics = evaluator.evaluate()

    println("Accuracy: %.2f%%".format(100.0 * stats.accuracy))
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING\n")

    SequenceTrainer(
      model = this.neuralNetwork,
      updateMethod = LearningRateMethod(learningRate = 0.1),
      lossCalculator = SoftmaxCrossEntropyCalculator,
      examples = this.dataset.training,
      epochs = 4,
      batchSize = 1,
      evaluator = SequenceEvaluator(
        model = this.neuralNetwork,
        examples = this.dataset.validation,
        outputEvaluationFunction = ClassificationEvaluation())
    ).train()
  }
}
