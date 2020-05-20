/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.GRUNeuralNetwork
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import traininghelpers.training.SequenceWithFinalOutputTrainer
import traininghelpers.validation.SequenceWithFinalOutputEvaluator
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.Corpus
import utils.SequenceExampleWithFinalOutput
import mnist.helpers.MNISTSequenceExampleExtractor
import utils.CorpusReader

fun main() {

  println("Start 'MNIST Sequence Test'")

  val dataset = CorpusReader<SequenceExampleWithFinalOutput<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist_sequence.datasets_paths, // same for validation and test
    exampleExtractor = MNISTSequenceExampleExtractor(outputSize = 10),
    perLine = true)

  MNISTSequenceTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
private class MNISTSequenceTest(val dataset: Corpus<SequenceExampleWithFinalOutput<DenseNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = GRUNeuralNetwork(
    inputSize = 2,
    hiddenSize = 200,
    hiddenActivation = Tanh,
    outputSize = 10,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {

    println("\n-- TRAINING")

    SequenceWithFinalOutputTrainer(
      model = this.neuralNetwork,
      updateMethod = RADAMMethod(stepSize = 0.001),
      lossCalculator = SoftmaxCrossEntropyCalculator,
      examples = this.dataset.training,
      epochs = 3,
      batchSize = 1,
      evaluator = SequenceWithFinalOutputEvaluator(
        model = this.neuralNetwork,
        examples = this.dataset.validation,
        outputEvaluationFunction = ClassificationEvaluation)
    ).train()
  }
}
