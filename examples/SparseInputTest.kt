/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Softsign
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import utils.Corpus
import utils.SimpleExample
import traininghelpers.training.FeedforwardTrainer
import traininghelpers.validation.FeedforwardEvaluator
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import utils.CorpusReader
import utils.exampleextractor.ClassificationSparseExampleExtractor

fun main() {

  println("Start 'Sparse Input Test'")

  val dataset = CorpusReader<SimpleExample<SparseBinaryNDArray>>().read(
    corpusPath = Configuration.loadFromFile().sparse_input.datasets_paths,
    exampleExtractor = ClassificationSparseExampleExtractor(inputSize = 356425, outputSize = 86),
    perLine = true)

  SparseInputTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
class SparseInputTest(val dataset: Corpus<SimpleExample<SparseBinaryNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = FeedforwardNeuralNetwork(
    inputSize = 356425,
    inputType = LayerType.Input.SparseBinary,
    hiddenSize = 200,
    hiddenActivation = Softsign,
    outputSize = 86,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {
    this.train()
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING")

    FeedforwardTrainer(
      model = this.neuralNetwork,
      updateMethod = AdaGradMethod(learningRate = 0.1),
      lossCalculator = SoftmaxCrossEntropyCalculator,
      examples = this.dataset.training,
      epochs = 3,
      batchSize = 1,
      evaluator = FeedforwardEvaluator(
        model = this.neuralNetwork,
        examples = this.dataset.validation,
        outputEvaluationFunction = ClassificationEvaluation)
    ).train()
  }
}
