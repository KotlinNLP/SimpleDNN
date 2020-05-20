/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attention.han.HAN
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attention.han.toHierarchySequence
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.Evaluator
import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.simplednn.helpers.Trainer
import com.kotlinnlp.utils.Shuffler
import utils.Corpus
import utils.SimpleExample
import utils.CorpusReader
import utils.exampleextractor.ClassificationExampleExtractor
import kotlin.math.roundToInt

/**
 * Train a HAN classifier for a sentiment classification (2 classes) and validate it.
 */
fun main() {

  println("Start 'HAN Classifier Test'")

  val dataset: Corpus<SimpleExample<DenseNDArray>> = CorpusReader<SimpleExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().han_classifier.datasets_paths, // same for validation and test
    exampleExtractor = ClassificationExampleExtractor(outputSize = 2),
    perLine = false)

  val trainingSize: Int = (0.9 * dataset.training.size).roundToInt()
  val trainingSet: List<SimpleExample<DenseNDArray>> = dataset.training.subList(0, trainingSize)
  val validationSet: List<SimpleExample<DenseNDArray>> = dataset.training.subList(trainingSize, dataset.training.size)

  val embeddings = EmbeddingsMap<Int>(size = 50)
  val model = HAN(
    hierarchySize = 1,
    inputSize = embeddings.size,
    inputType = LayerType.Input.Dense,
    biRNNsActivation = Tanh,
    biRNNsConnectionType = LayerType.Connection.RAN,
    attentionSize = embeddings.size,
    outputSize = 2,
    outputActivation = Softmax())

  println("\n-- TRAINING")
  println("Using %d/%d examples as training set and %d/%d as validation set.".format(
    trainingSet.size, dataset.training.size, validationSet.size, dataset.training.size))

  HANClassifierTrainer(
    model = model,
    examples = trainingSet,
    embeddings = embeddings,
    evaluator = HANClassifierEvaluator(model = model, examples = validationSet, embeddings = embeddings),
    updateMethod = RADAMMethod(stepSize = 0.001)
  ).train()

  println("\n-- TEST")
  val stats: Statistics =
    HANClassifierEvaluator(model = model, examples = dataset.test, embeddings = embeddings).evaluate()
  println("Accuracy: %.2f%%".format(100.0 * stats.accuracy))
}

/**
 * The HAN classifier trainer.
 */
private class HANClassifierTrainer(
  model: HAN,
  examples: List<SimpleExample<DenseNDArray>>,
  private val embeddings: EmbeddingsMap<Int>,
  evaluator: HANClassifierEvaluator,
  updateMethod: UpdateMethod<*>
) : Trainer<SimpleExample<DenseNDArray>>(
  modelFilename = "",
  optimizers = listOf(ParamsOptimizer(updateMethod)),
  examples = examples,
  epochs = 10,
  batchSize = 1,
  evaluator = evaluator,
  shuffler = Shuffler(),
  verbose = true
) {

  /**
   * The [HANEncoder] used as classifier (Softmax output activation).
   */
  private val classifier = HANEncoder<DenseNDArray>(model = model, useDropout = false, propagateToInput = false)

  /**
   * Learn from an example (forward + backward).
   *
   * @param example the example used to train the network
   */
  override fun learnFromExample(example: SimpleExample<DenseNDArray>) {

    val inputSequence: Array<DenseNDArray> = extractInputSequence(example = example, embeddings = this.embeddings)
    val output: DenseNDArray = this.classifier.forward(inputSequence.toHierarchySequence())

    this.classifier.backward(outputErrors = output.assignSub(example.outputGold))
  }

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromExample].
   */
  override fun accumulateErrors() {
    this.optimizers.single().accumulate(this.classifier.getParamsErrors(copy = false))
  }

  /**
   * Dump the model to file.
   */
  override fun dumpModel() {}
}

/**
 * The HAN classifier evaluator.
 *
 * @param model the model to validate
 * @param examples a list of examples to validate
 */
private class HANClassifierEvaluator(
  model: HAN,
  examples: List<SimpleExample<DenseNDArray>>,
  private val embeddings: EmbeddingsMap<Int>
) : Evaluator<SimpleExample<DenseNDArray>, Statistics.Simple>(
  examples = examples,
  verbose = true
) {

  /**
   * The validation statistics.
   */
  override val stats: Statistics.Simple = Statistics.Simple()

  /**
   * The [HANEncoder] for the classification.
   */
  private val classifier = HANEncoder<DenseNDArray>(model = model, propagateToInput = false, useDropout = false)

  /**
   * Evaluate the model with a single example.
   *
   * @param example the example to validate the model with
   */
  override fun evaluate(example: SimpleExample<DenseNDArray>) {

    val inputSequence: Array<DenseNDArray> = extractInputSequence(example = example, embeddings = this.embeddings)
    val output: DenseNDArray = this.classifier.forward(inputSequence.toHierarchySequence())

    if (ClassificationEvaluation(output = output, outputGold = example.outputGold))
      this.stats.metric.truePos++
    else
      this.stats.metric.falsePos++

    this.stats.accuracy = this.stats.metric.precision
  }
}

/**
 * @param example an example of the dataset
 * @param embeddings the input embeddings
 *
 * @return an array of embeddings vectors associated to each feature (casted to Int) of the [example]
 */
private fun extractInputSequence(example: SimpleExample<DenseNDArray>,
                                 embeddings: EmbeddingsMap<Int>): Array<DenseNDArray> =
  example.features
    .toDoubleArray()
    .map { embeddings.getOrSet(it.toInt()).values }
    .toTypedArray()
