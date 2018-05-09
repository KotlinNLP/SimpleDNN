/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [PointerNetwork].
 *
 * @property model the model of the network
 */
class PointerNetwork(val model: PointerNetworkModel) {

  /**
   * The size of the currently processing sequence (set with the [setInputSequence] method).
   */
  val sequenceSize: Int get() = this.inputSequence.size

  /**
   * The processor of the recurrent network.
   */
  val recurrentProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(this.model.recurrentNetwork)

  /**
   * A boolean indicating if the current is the first state.
   */
  private var firstState: Boolean = true

  /**
   * The input sequence that must be set using the [setInputSequence] method.
   */
  private lateinit var inputSequence: List<DenseNDArray>

  /**
   * The init hidden array (null by default).
   */
  private var initHidden: DenseNDArray? = null

  /**
   * The forward helper.
   */
  private val forwardHelper = ForwardHelper(network = this)

  /**
   * The backward helper.
   */
  private val backwardHelper = BackwardHelper(network = this)

  /**
   * Set the input sequence.
   *
   * @param inputSequence the input sequence
   */
  fun setInputSequence(inputSequence: List<DenseNDArray>) {

    this.firstState = true
    this.inputSequence = inputSequence
    this.initHidden = null
  }

  /**
   * Set the init hidden array.
   *
   * @param array the init hidden array
   */
  fun setInitHidden(array: DenseNDArray) {
    this.initHidden = array
  }

  /**
   * Forward.
   *
   * @return the output array of the network
   */
  fun forward(): DenseNDArray {

    val output: DenseNDArray = this.forwardHelper.forward(
      inputSequence = this.inputSequence,
      firstState = this.firstState,
      initHidden = this.initHidden)

    this.firstState = false

    return output
  }

  /**
   * Back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: List<DenseNDArray>) {

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
  fun getInputSequenceErrors(): List<DenseNDArray> = this.backwardHelper.inputSequenceErrors

  /**
   * Get the errors of the init hidden array.
   * They can be get only if the forward has been called with an init hidden array.
   *
   * @return the errors of the init hidden array
   */
  fun getInitHiddenErrors(): DenseNDArray = this.backwardHelper.initHiddenErrors
}
