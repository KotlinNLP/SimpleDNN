/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.simplednn.helpers.Trainer
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.WordPieceTokenizer
import java.io.File
import java.io.FileOutputStream
import java.util.*

/**
 * The trainer of a [BERT] model.
 *
 * @param model the model to train
 * @param modelFilename the name of the file in which to save the serialized model
 * @param termsDropout the probability to dropout an input token
 * @param optimizeEmbeddings whether to optimize the embeddings during the training
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param examples the training examples
 * @param epochs the number of training epochs
 * @param shuffler shuffle the examples before each epoch, converting the sequence to a list
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class BERTTrainer(
  private val model: BERTModel,
  modelFilename: String,
  private val termsDropout: Double = 0.15,
  private val optimizeEmbeddings: Boolean,
  updateMethod: UpdateMethod<*>,
  examples: Iterable<String>,
  epochs: Int,
  shuffler: Shuffler? = Shuffler(),
  verbose: Boolean = true
) : Trainer<String>(
  modelFilename = modelFilename,
  optimizers = listOf(ParamsOptimizer(updateMethod)),
  examples = examples,
  epochs = epochs,
  batchSize = 1,
  evaluator = null,
  shuffler = shuffler,
  verbose = verbose
) {

  companion object {

    /**
     * The random generator used to decide if an token must be masked.
     */
    private val randomGenerator = Random(743)

    /**
     * The special tokens that must not be split with the tokenization.
     */
    private val SPECIAL_TOKENS: Set<String> = BERTModel.FuncToken.values().map { it.form }.toSet()
  }

  /**
   * The encoded term of an example.
   *
   * @property form the term form
   */
  private inner class EncodedTerm(val form: String) {

    /**
     * The BERT encoding.
     */
    lateinit var encoding: DenseNDArray

    /**
     * The classification index of this term within the vocabulary.
     * Note: the unknown terms are not considered for the classification.
     */
    val index: Int = this@BERTTrainer.model.vocabulary.getId(this.form) ?: -1

    /**
     * Whether this term must be considered as a masked input.
     */
    val isMasked: Boolean =
      this.form in this@BERTTrainer.model.vocabulary && randomGenerator.nextDouble() < this@BERTTrainer.termsDropout
  }

  /**
   * The examples tokenizer.
   */
  private val tokenizer = WordPieceTokenizer(this.model.vocabulary)

  /**
   * A Bidirectional Encoder Representations from Transformers.
   */
  private val bert = BERT(
    model = this.model,
    useDropout = true,
    propagateToInput = this.optimizeEmbeddings,
    masksEnabled = true)

  /**
   * A feed-forward layer trained to classify an encoded vector within the terms of the model vocabulary.
   * It is used only during the training phase.
   */
  private val classifier: FeedforwardNeuralProcessor<DenseNDArray> =
    FeedforwardNeuralProcessor(model = this.model.classifier, propagateToInput = true, useDropout = false)

  /**
   * The errors given when a term has not been dropped.
   */
  private val zeroErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.inputSize))

  /**
   * The classification stats.
   */
  private val stats = Statistics.Simple()

  /**
   * The losses of the last classifications made.
   */
  private val lastLosses: MutableList<Double> = mutableListOf()

  /**
   * The examples iteration counter.
   */
  private var examplesCount = 0

  /**
   * A timer to track the elapsed time.
   */
  private val timer = Timer()

  /**
   * Check requirements.
   */
  init {
    require(this.termsDropout.let { it > 0.0 && it < 1.0 }) { "The terms dropout must be in the range (0.0, 1.0)" }
  }

  /**
   * Learn from an example (forward + backward).
   *
   * @param example an example to train the model with
   */
  override fun learnFromExample(example: String) {

    val forms: List<String> = this.tokenizer.tokenize(example, neverSplit = SPECIAL_TOKENS)
    val encodedTerms: List<EncodedTerm> = forms.map { EncodedTerm(it) }

    this.bert.forward(encodedTerms.map { if (it.isMasked) BERTModel.FuncToken.MASK.form else it.form })
      .zip(encodedTerms)
      .forEach { (encoding, encodedTerm) -> encodedTerm.encoding = encoding }

    val encodingErrors: List<DenseNDArray> = encodedTerms.map {
      if (it.isMasked)
        this.classifyVector(vector = it.encoding, goldIndex = it.index)
      else
        this.zeroErrors
    }

    this.bert.backward(encodingErrors)

    if (this.verbose) this.printProgressAndStats()
  }

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromExample].
   */
  override fun accumulateErrors() {
    this.optimizers.single().accumulate(this.bert.getParamsErrors(copy = false), copy = false)
  }

  /**
   * Dump the model to file.
   */
  override fun dumpModel() {
    this.model.dump(FileOutputStream(File(this.modelFilename)))
  }

  /**
   * Classify a vector (representing a term) comparing the result with the expected term index.
   *
   * @param vector the vector to classify
   * @param goldIndex the index of the classifying term
   *
   * @return the vector errors respect to the classification made
   */
  private fun classifyVector(vector: DenseNDArray, goldIndex: Int): DenseNDArray {

    this.classifier.forward(vector)

    val classification: DenseNDArray = this.classifier.getOutput(copy = false)
    val goldOutput: DenseNDArray =
      DenseNDArrayFactory.oneHotEncoder(length = this.classifier.model.outputSize, oneAt = goldIndex)

    this.updateStats(classification = classification, goldOutput = goldOutput)

    this.classifier.backward(
      SoftmaxCrossEntropyCalculator.calculateErrors(output = classification, outputGold = goldOutput))
    this.optimizers.single().accumulate(this.classifier.getParamsErrors(copy = false))

    return this.classifier.getInputErrors(copy = true)
  }

  /**
   * Update the classification stats.
   *
   * @param classification the last classification made
   * @param goldOutput the expected classification
   */
  private fun updateStats(classification: DenseNDArray, goldOutput: DenseNDArray) {

    val predictedIndex: Int = classification.argMaxIndex()
    val goldIndex: Int = goldOutput.argMaxIndex()

    this.lastLosses.add(
      SoftmaxCrossEntropyCalculator.calculateLoss(output = classification, outputGold = goldOutput).sum())

    if (predictedIndex == goldIndex)
      this.stats.metric.truePos++
    else
      this.stats.metric.falsePos++
  }

  /**
   * Print progress and stats.
   */
  private fun printProgressAndStats() {

    this.examplesCount++

    if (this.examplesCount % 100 == 0)
      print(".")

    if (this.examplesCount % 1000 == 0) {

      val lossStr = "loss %.2f".format(this.lastLosses.average())
      println("\n[${this.timer.formatElapsedTime()}] After $examplesCount examples: $lossStr | ${this.stats.metric}")

      this.validateAndSaveModel()
      this.lastLosses.clear()
      this.stats.reset()
    }
  }
}
