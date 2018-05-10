/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [PointerNetwork].
 *
 * @property model the model of the network
 */
class PointerNetwork(val model: PointerNetworkModel) {

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
   * The forward helper.
   */
  internal val forwardHelper = ForwardHelper(network = this)

  /**
   * The backward helper.
   */
  private val backwardHelper = BackwardHelper(network = this)

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
   * @param vector the vector that modulates a content-based attention mechanism over the input sequence
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(vector: DenseNDArray): DenseNDArray {

    val output: DenseNDArray = this.forwardHelper.forward(vector)

    this.forwardCount++

    return output
  }

  /**
   * Back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    require(outputErrors.size == this.forwardCount)

    this.backwardHelper.backward(outputErrors = outputErrors)
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of this network
   */
  fun getParamsErrors(copy: Boolean = true): PointerNetworkParameters =
    this.backwardHelper.getParamsErrors(copy = copy)

  /**
   * @return the errors of the sequence
   */
  fun getInputErrors() = InputErrors(
    inputSequenceErrors = this.backwardHelper.inputSequenceErrors,
    inputVectorsErrors = this.backwardHelper.vectorsErrors)
}
