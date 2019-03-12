/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.core.attention.AttentionMechanism
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool

/**
 * The [PointerNetworkProcessor].
 *
 * @property model the model of the network
 * @property id an identification number useful to track a specific processor
 */
class PointerNetworkProcessor(
  val model: PointerNetworkModel,
  override val id: Int = 0
) : NeuralProcessor<
  DenseNDArray, // InputType
  DenseNDArray, // OutputType
  List<DenseNDArray>, // ErrorsType
  PointerNetworkProcessor.InputErrors, // InputErrorsType
  PointerNetworkParameters // ParamsType
  > {

  /**
   * Propagate to the input by default
   */
  override val propagateToInput: Boolean = true

  /**
   * TODO: implement the dropout
   */
  override val useDropout: Boolean = false

  /**
   * @param inputSequenceErrors the list of errors of the input sequence
   * @param inputVectorsErrors The list of errors of the input vectors
   */
  class InputErrors(
    val inputSequenceErrors: List<DenseNDArray>,
    val inputVectorsErrors: List<DenseNDArray>)

  /**
   * The input sequence that must be set using the [setInputSequence] method.
   */
  internal lateinit var inputSequence: List<DenseNDArray>

  /**
   * The number of forwards performed during the last decoding.
   */
  internal var forwardCount: Int = 0

  /**
   * A boolean indicating if this is the first state.
   */
  internal val firstState: Boolean get() = this.forwardCount == 0

  /**
   * A pool of processors for the merge network.
   */
  internal val mergeProcessorsPool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
    model = this.model.mergeNetwork,
    useDropout = false,
    propagateToInput = true)

  /**
   * The list of merge processors used during the last forward.
   */
  internal val usedMergeProcessors = mutableListOf<List<FeedforwardNeuralProcessor<DenseNDArray>>>()

  /**
   * The list of attention mechanisms used during the last forward.
   */
  internal val usedAttentionMechanisms = mutableListOf<AttentionMechanism>()

  /**
   * The forward helper.
   */
  private val forwardHelper = ForwardHelper(networkProcessor = this)

  /**
   * The backward helper.
   */
  private val backwardHelper: BackwardHelper by lazy { BackwardHelper(networkProcessor = this) }

  /**
   * Set the encoded sequence.
   *
   * @param inputSequence the input sequence
   */
  fun setInputSequence(inputSequence: List<DenseNDArray>) {

    this.forwardCount = 0
    this.inputSequence = inputSequence
  }

  /**
   * Forward.
   *
   * @param input the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  override fun forward(input: DenseNDArray): DenseNDArray {

    val output: DenseNDArray = this.forwardHelper.forward(input)

    this.forwardCount++

    return output
  }

  /**
   * @return an array that contains the importance scores NOT activated resulting from the last forward
   */
  fun getLastNotActivatedImportanceScores(): DenseNDArray =
    this.usedAttentionMechanisms.last().outputArray.valuesNotActivated.copy()

  /**
   * Back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    require(outputErrors.size == this.forwardCount)

    this.backwardHelper.backward(outputErrors = outputErrors)
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference (default true)
   *
   * @return the params errors of this network
   */
  override fun getParamsErrors(copy: Boolean): PointerNetworkParameters =
    this.backwardHelper.getParamsErrors(copy = copy)

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean) = InputErrors(
    inputSequenceErrors = this.backwardHelper.inputSequenceErrors,
    inputVectorsErrors = this.backwardHelper.vectorsErrors
  )
}
