/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.linguisticdescription.sentence.flattenTokens
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.simplednn.helpers.Trainer
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.Timer
import java.io.File
import java.io.FileOutputStream

/**
 * The trainer of a [BERT] model.
 *
 * @param model the parameters of the model of the network
 * @param modelFilename the name of the file in which to save the serialized model
 * @param tokenizer a neural tokenizer
 * @param embeddingsMap pre-trained word embeddings
 * @param dictionary a dictionary set with all the forms in the examples
 * @param termsDropout the probability to dropout an input token
 * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 * @param examples the training examples
 * @param epochs the number of training epochs
 * @param shuffler shuffle the examples before each epoch, converting the sequence to a list
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class BERTTrainer(
  private val model: BERTParameters,
  modelFilename: String,
  private val tokenizer: NeuralTokenizer,
  private val embeddingsMap: EmbeddingsMap<String>,
  private val dictionary: DictionarySet<String>,
  private val termsDropout: Double = 0.15,
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

  /**
   * The encoded term of an example.
   *
   * @property form the term form
   * @property embedding the term embedding
   * @property dropped whether this term has been dropped in input
   */
  private data class EncodedTerm(val form: String, val embedding: ParamsArray, val dropped: Boolean) {

    /**
     * The BERT encoding.
     */
    lateinit var encoding: DenseNDArray
  }

  /**
   * A Bidirectional Encoder Representations from Transformers.
   */
  private val bert = BERT(model = this.model, useDropout = true)

  /**
   * A feed-forward layer trained to classify an encoded vector within the terms of the dictionary.
   * It is used only during the training phase.
   */
  private val classificationLayer: FeedforwardLayer<DenseNDArray> = FeedforwardLayer(
    inputArray = AugmentedArray.zeros(size = this.model.inputSize),
    outputArray = AugmentedArray.zeros(size = this.dictionary.size + 1), // dictionary size + unknown
    params = FeedforwardLayerParameters(inputSize = this.model.inputSize, outputSize = this.dictionary.size + 1),
    inputType = LayerType.Input.Dense,
    activationFunction = Softmax())

  /**
   * The errors given when a term has not been dropped.
   */
  private val zeroErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.inputSize))

  /**
   * The index of the unknown term in the classification.
   */
  private val unknownIndex: Int = this.dictionary.size

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
   * Learn from an example (forward + backward).
   *
   * @param example an example to train the model with
   */
  override fun learnFromExample(example: String) {

    val forms: List<String> = this.tokenizer.tokenize(example).flattenTokens().map { it.form }
    val encodedTerms: List<EncodedTerm> = this.encodeExample(forms)

    this.bert.forward(encodedTerms.map { it.embedding.values })
      .zip(encodedTerms)
      .forEach { (encoding, encodedTerm) -> encodedTerm.encoding = encoding }

    val encodingErrors: List<DenseNDArray> = encodedTerms.map {
      if (it.dropped)
        this.classifyVector(vector = it.encoding, goldIndex = this.getId(it.form))
      else
        this.zeroErrors
    }

    this.bert.backward(encodingErrors)

    if (this.verbose) this.printProgressAndStats()
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

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromExample].
   */
  override fun accumulateErrors() {
    this.optimizers.single().accumulate(this.bert.getParamsErrors(copy = false))
  }

  /**
   * Dump the model to file.
   */
  override fun dumpModel() {
    this.model.dump(FileOutputStream(File(this.modelFilename)))
  }

  /**
   * @param forms the forms that compose an example
   *
   * @return the encoded terms
   */
  private fun encodeExample(forms: List<String>): List<EncodedTerm> = forms.map { form ->

    val embedding = this.embeddingsMap.get(key = form, dropout = this.termsDropout)
    val dropped: Boolean = this.embeddingsMap.contains(form) && embedding == this.embeddingsMap.unknownEmbedding

    EncodedTerm(form = form, embedding = embedding, dropped = dropped)
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

    this.classificationLayer.setInput(vector)
    this.classificationLayer.forward()

    val classification: DenseNDArray = this.classificationLayer.outputArray.values
    val goldOutput: DenseNDArray =
      DenseNDArrayFactory.oneHotEncoder(length = this.classificationLayer.params.outputSize, oneAt = goldIndex)
    val errors: DenseNDArray = classification.sub(goldOutput)

    this.updateStats(classification = classification, goldOutput = goldOutput)

    this.classificationLayer.setErrors(errors)
    this.optimizers.single().accumulate(this.classificationLayer.backward(propagateToInput = true))

    return this.classificationLayer.inputArray.errors.copy()
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
      SoftmaxCrossEntropyCalculator().calculateLoss(output = classification, outputGold = goldOutput).sum())

    when (predictedIndex) {
      goldIndex -> if (goldIndex != this.unknownIndex) this.stats.metric.truePos++
      this.unknownIndex -> this.stats.metric.falseNeg
      else -> this.stats.metric.falsePos++
    }
  }

  /**
   * @param form a form found in the examples
   *
   * @return a unique ID of the given form, considering all the unknown terms equal to each other
   */
  private fun getId(form: String): Int =
    this.dictionary.getId(form) ?: this.unknownIndex
}
