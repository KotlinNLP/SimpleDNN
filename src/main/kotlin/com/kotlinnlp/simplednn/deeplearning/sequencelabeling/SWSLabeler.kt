/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequencelabeling

import com.kotlinnlp.simplednn.core.functionalities.losses.MulticlassMSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The [SWSLabeler] uses a neural network to assign a categorical label to each member of a sequence.
 *
 * Note: The SWSLabeler can can process one sequence as time therefore is not thread-safe.
 *
 * @param network the [SWSLNetwork] of this encoder
 * @param useDropout whether to apply the dropout during the forward
 */
class SWSLabeler(
  private val network: SWSLNetwork,
  private val useDropout: Boolean = false
) {

  /**
   * @property paramsErrors the errors on the network parameters
   * @property labelsEmbeddingsErrors the errors on the labels embeddings
   * @param inputErrors the errors on the input elements
   */
  data class NetworkErrors(
    val paramsErrors: NetworkParameters,
    val labelsEmbeddingsErrors: DenseNDArray,
    val inputErrors: DenseNDArray)

  /**
   * The output predicted label with the prediction score.
   *
   * @property index the outcome index to which this label is associated
   * @property score the prediction score (Double in range [0.0, 1.0], default = 1.0)
   */
  data class Label(val index: Int, val score: Double = 1.0)

  /**
   * The Sliding Window Sequence which is being processed.
   */
  private lateinit var sequence: SlidingWindowSequence

  /**
   * The list of labels, parallel to the current sequence.
   * The max size is the length of the current sequence.
   */
  private val labels = mutableListOf<Label>()

  /**
   * The feed-forward neural processor.
   */
  private val processor = FeedforwardNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.network.classifier,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The loss calculator used to calculate the loss between the expected output and the predicted output.
   */
  private val lossCalculator = MulticlassMSECalculator()

  /**
   * The input errors calculated during the back-propagation.
   */
  private lateinit var inputSequenceErrors: List<DenseNDArray>

  /**
   * Annotate each element of the given [inputSequence] with a label.
   *
   * @param inputSequence the input sequence to annotate
   *
   * @return an array of [Label]s, one for each element of the [inputSequence]
   */
  fun annotate(inputSequence: List<DenseNDArray>): List<Label> {

    require(inputSequence.isNotEmpty())

    this.setNewSequence(inputSequence)

    this.forwardSequence(forEachPrediction = { this.addLabel(this.getBestLabel()) } )

    return this.labels
  }

  /**
   * Train the sequence labeler.
   *
   * @param inputSequence the input sequence to annotate
   * @param goldLabels the expected labels for each element
   * @param optimizer the optimize of parameters and embeddings
   */
  fun learn(inputSequence: List<DenseNDArray>,
            goldLabels: IntArray,
            optimizer: SWSLOptimizer) {

    require(inputSequence.isNotEmpty())

    this.setNewSequence(inputSequence)
    this.initInputErrors(inputSequence.size)

    this.forwardSequence(
      forEachPrediction = {

        val goldLabel = this.getGoldLabel(goldLabels)

        this.processor.backward(outputErrors = this.getOutputErrors(goldLabel))

        this.accumulateErrors(optimizer)

        this.addLabel(goldLabel)
      }
    )
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the input errors
   */
  fun getInputSequenceErrors(copy: Boolean = true): List<DenseNDArray> =
    this.inputSequenceErrors.let { if (copy) it.map { it.copy() } else it }

  /**
   * Set a new Sliding Window Sequence initialized with the elements of [inputSequence].
   * Clear the labels and the input errors.
   *
   * @param inputSequence the inputSequence to annotate
   */
  private fun setNewSequence(inputSequence: List<DenseNDArray>) {

    this.sequence = SlidingWindowSequence(
      elements = inputSequence,
      leftContextSize = this.network.leftContextSize,
      rightContextSize = this.network.rightContextSize)

    this.labels.clear()
  }

  /**
   * Initialize input sequence errors.
   */
  private fun initInputErrors(size: Int) {
    this.inputSequenceErrors = List(size = size, init = { DenseNDArrayFactory.zeros(Shape(this.network.elementSize)) })
  }

  /**
   * Add the given [Label] to [labels].
   *
   * @param label a [Label]
   */
  private fun addLabel(label: Label) = this.labels.add(label)

  /**
   * @return the predicted [Label] of the last processed element
   */
  private fun getBestLabel(): Label {

    val output = this.processor.getOutput(copy = false)
    val bestIndex = output.argMaxIndex()

    return Label(index = bestIndex, score = output[bestIndex])
  }

  /**
   * @return the gold [Label] of the focus element of the sequence
   */
  private fun getGoldLabel(goldLabels: IntArray) = Label(index = goldLabels[this.sequence.focusIndex], score = 1.0)

  /**
   * This function iterates the complete sequence to make a prediction for each element.
   *
   * @param forEachPrediction the callback called after the forward on each element
   */
  private fun forwardSequence(forEachPrediction: () -> Unit) {

    val featuresExtractor = SWSLFeaturesExtractor(
      sequence = this.sequence,
      labels = this.labels,
      network = this.network)

    while (this.sequence.focusInRange()) {

      this.processor.forward(featuresExtractor.getFeatures())

      forEachPrediction()

      this.sequence.shift()
    }
  }

  /**
   * Calculate the errors of the last prediction respect to the [goldLabel].
   *
   * @param goldLabel the gold [Label] of the last prediction
   *
   * @return the errors
   */
  private fun getOutputErrors(goldLabel: Label) = lossCalculator.calculateErrors(
    output = this.processor.getOutput(copy = false),
    outputGold = DenseNDArrayFactory.oneHotEncoder(length = this.network.numberOfLabels, oneAt = goldLabel.index))

  /**
   * Propagate the last backward errors to the network parameters (params and label embeddings) and to the input
   * elements.
   *
   * @param optimizer the optimizer in which to accumulate errors
   */
  private fun accumulateErrors(optimizer: SWSLOptimizer) {

    val networErrors = this.getNetworkErrors()

    optimizer.accumulateParamsErrors(networErrors.paramsErrors)
    this.accumulateLabelsEmbeddingsErrors(networErrors.labelsEmbeddingsErrors, optimizer = optimizer)
    this.accumulateInputErrors(networErrors.inputErrors)
  }

  /**
   * @return the errors on the params, label embeddings end input elements
   */
  private fun getNetworkErrors(): NetworkErrors {

    val inputLayerErrors: DenseNDArray = this.processor.getInputErrors(copy = false)

    return NetworkErrors(
      paramsErrors = this.processor.getParamsErrors(copy = false),
      labelsEmbeddingsErrors = inputLayerErrors.getRange(0, this.network.labelsEmbeddingsSize),
      inputErrors = inputLayerErrors.getRange(this.network.labelsEmbeddingsSize, this.network.featuresSize))
  }

  /**
   * Accumulate the given labels embeddings errors.
   *
   * @param errors the errors to accumulate
   * @param optimizer the optimizer in which to accumulate [errors]
   */
  private fun accumulateLabelsEmbeddingsErrors(errors: DenseNDArray, optimizer: SWSLOptimizer) {

    this.alignLabelsEmbeddingsErrors(errors).forEach { (embeddingIndex, embeddingErrors) ->
      optimizer.accumulateLabelEmbeddingErrors(embeddingId = embeddingIndex, errors = embeddingErrors)
    }
  }

  /**
   * Accumulate the given input sequence errors.
   *
   * @param errors the errors to accumulate
   */
  private fun accumulateInputErrors(errors: DenseNDArray) {

    this.sequence.getContext().zip(errors.splitV(this.network.elementSize)).forEach { (i, e) ->

      if (i != null) this.inputSequenceErrors[i].assignSum(e)
    }
  }

  /**
   * Align the errors on the labels embeddings to the current labels.
   *
   * @param errors the errors to align with the current labels
   *
   * @return a list of pair <labelIndex, embeddingErrors>
   */
  private fun alignLabelsEmbeddingsErrors(errors: DenseNDArray): List<Pair<Int, DenseNDArray>> {

    val result = mutableListOf<Pair<Int, DenseNDArray>>()

    if (this.labels.isNotEmpty()) {

      val splitErrors = errors.splitV(this.network.labelEmbeddingSize).reversed()

      val firstIndex = maxOf(0, this.labels.size - splitErrors.size)

      (this.labels.lastIndex downTo firstIndex).mapIndexedTo(result) { k, i ->
        Pair(this.labels[i].index, splitErrors[k])
      }
    }

    return result
  }
}
