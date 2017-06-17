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
import java.util.*

/**
 * The [SWSLabeler] uses a neural network to assign a categorical label to each member of a sequence.
 *
 * Note: The SWSLabeler can can process one sequence as time therefore is not thread-safe.
 *
 * @property network the [SWSLNetwork] of this encoder
 * @property optimizer the optimizer associated to the [SWSLNetwork] (can be null)
 */
@Suppress("unused")
class SWSLabeler(
  private val network: SWSLNetwork,
  private val optimizer: SWSLOptimizer?){

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
   * The Sliding Window Sequence which is being processed
   */
  private var sequence: SlidingWindowSequence? = null

  /**
   * The list of labels parallel to the current sequence.
   * The max size is the length of the current sequence
   */
  private val labels = ArrayList<Int>()

  /**
   * The feed-forward neural processor
   */
  private val processor = FeedforwardNeuralProcessor<DenseNDArray>(neuralNetwork = this.network.classifier)

  /**
   * The loss calculator used to calculate the loss between
   * the expected output and the predicted output
   */
  private val lossCalculator = MulticlassMSECalculator()

  /**
   * The input errors calculated during the back-propagation
   */
  private var inputSequenceErrors: Array<DenseNDArray>? = null

  /**
   * This is the main function to annotate the input [elements] with labels
   *
   * @param elements the input sequence to annotate
   *
   * @return an array of labels of the same size of [elements]
   */
  @Suppress("unused")
  fun annotate(elements: Array<DenseNDArray>): ArrayList<Int> {

    this.setNewSequence(elements)

    this.processSequence({ this.addLabel(this.getBestLabel()) })

    return this.labels
  }

  /**
   * This is the main function to train the sequence labeler
   *
   * @param elements the input sequence to annotate
   * @param goldLabels the expected labels for each element
   */
  @Suppress("unused")
  fun learn(elements: Array<DenseNDArray>, goldLabels: IntArray){

    require(optimizer != null) { "Impossible to learn without an optimizer" }

    this.setNewSequence(elements)

    this.processSequence({

      val goldLabel = this.getGoldLabel(goldLabels)

      this.processor.backward(
        outputErrors = this.getOutputErrors(goldLabel),
        propagateToInput = true)

      this.propagateErrors()

      this.addLabel(goldLabel)
    })
  }

  /**
   * @return the input errors
   */
  @Suppress("unused")
  fun getInputSequenceErrors(): Array<DenseNDArray> = this.inputSequenceErrors!!

  /**
   * Set a new Sliding Window Sequence initialized with [elements]
   * Clear the labels and the input errors.
   *
   * @param elements the elements of the sequence
   */
  private fun setNewSequence(elements: Array<DenseNDArray>){

    this.sequence = SlidingWindowSequence(
      elements = elements,
      leftContextSize = this.network.leftContextSize,
      rightContextSize = this.network.rightContextSize)

    this.labels.clear()

    this.inputSequenceErrors = Array(size = elements.size, init = {
      DenseNDArrayFactory.zeros(Shape(this.network.elementSize))
    })
  }

  /**
   * Assign a label to the current sequence focus element
   */
  private fun addLabel(label: Int) = this.labels.add(label)

  /**
   * @return the neural network best guess label for last the processed element
   */
  private fun getBestLabel(): Int = this.processor.getOutput(copy = false).argMaxIndex()

  /**
   * @return the gold label for the focus element of the sequence
   */
  private fun getGoldLabel(goldLabels: IntArray): Int = goldLabels[this.sequence!!.focusIndex]

  /**
   * This function iterates the complete sequence to make a prediction
   * for each element using the [FeedforwardNeuralProcessor].
   *
   * The [callback] allows you to intercept the status of the neural processor
   * after the forward on each element
   *
   * @param callback the function to invoke to read the intermediate neural processor results
   */
  private fun processSequence(callback: () -> Unit) {

    val featuresExtractor = SWSLFeaturesExtractor(
      sequence = this.sequence!!,
      labels = this.labels,
      network = this.network)

    while (this.sequence!!.hasNext()){

      this.sequence!!.shift()

      this.processor.forward(featuresExtractor.getFeatures())

      callback()
    }
  }

  /**
   * Calculate the errors of the last prediction respect to the [goldLabel]
   *
   * @param goldLabel a gold label
   *
   * @return the errors
   */
  private fun getOutputErrors(goldLabel: Int) = lossCalculator.calculateErrors(
    output = this.processor.getOutput(copy = false),
    outputGold = DenseNDArrayFactory.oneHotEncoder(length = this.network.numberOfLabels, oneAt = goldLabel))

  /**
   * Propagate the last backward errors to the network parameters
   * (params and label embeddings) and to the input elements
   */
  private fun propagateErrors(){

    val (paramsErrors, labelsEmbeddingsErrors, inputErrors) = this.getNetworkErrors()

    this.propagateParamsErrors(paramsErrors)
    this.propagateLabelsEmbeddingsErrors(labelsEmbeddingsErrors)
    this.propagateInputErrors(inputErrors)
  }

  /**
   * @return the errors on the params, label embeddings end input elements
   */
  private fun getNetworkErrors(): NetworkErrors {

    val inputLayerErrors: DenseNDArray = this.processor.getInputErrors(copy = false)

    return NetworkErrors(
      paramsErrors = this.processor.getParamsErrors(copy = false),
      labelsEmbeddingsErrors = inputLayerErrors.getRange(0, this.network.labelsEmbeddingsSize),
      inputErrors = inputLayerErrors.getRange(this.network.labelsEmbeddingsSize, this.network.featuresSize)
    )
  }

  /**
   * Propagate the [paramsErrors] to the network parameters
   *
   * @param paramsErrors the params errors to propagate
   */
  private fun propagateParamsErrors(paramsErrors: NetworkParameters){
    this.optimizer!!.accumulateErrors(paramsErrors)
  }

  /**
   * Propagate the [errors] to the input elements
   *
   * @param errors the errors to propagate
   */
  private fun propagateInputErrors(errors: DenseNDArray){

    this.sequence!!.getContext().zip(errors.splitV(this.network.elementSize)).forEach { (i, e) ->

      if (i != null) this.inputSequenceErrors!![i].assignSum(e)
    }
  }

  /**
   * Propagate the [errors] to the labels embeddings
   *
   * @param errors the errors to propagate
   */
  private fun propagateLabelsEmbeddingsErrors(errors: DenseNDArray){

    this.alignLabelsEmbeddingsErrors(errors).forEach { (embeddingIndex, embeddingErrors) ->

      this.optimizer!!.accumulateLabelEmbeddingErrors(embeddingIndex = embeddingIndex, errors = embeddingErrors)
    }
  }

  /**
   * Align the errors on the labels embeddings to the current labels
   *
   * @param errors the errors to align with the current labels
   *
   * @return a list of pair <embeddingIndex, embeddingErrors>
   */
  private fun alignLabelsEmbeddingsErrors(errors: DenseNDArray): ArrayList<Pair<Int, DenseNDArray>>{

    val result = ArrayList<Pair<Int, DenseNDArray>>()

    if (this.labels.isNotEmpty()) {

      val splitErrors = errors.splitV(splittingLength = this.network.labelEmbeddingSize).reversed()

      val firstIndex = maxOf(0, this.labels.size - splitErrors.size)

      (this.labels.lastIndex downTo firstIndex).mapIndexedTo(result) { k, i ->
        Pair(this.labels[i], splitErrors[k])
      }
    }

    return result
  }
}