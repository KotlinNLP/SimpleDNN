/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The backward helper of the [PointerNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class BackwardHelper(private val network: PointerNetwork) {

  /**
   * The list of errors of the current input sequence.
   */
  lateinit var inputSequenceErrors: List<DenseNDArray>
    private set

  /**
   * The errors of the init hidden array. They can be get only if the forward has been called with an init hidden array.
   */
  val initHiddenErrors: DenseNDArray get() = this.network.recurrentProcessor.getInitHiddenErrors().first()!!

  /**
   * The index of the current state (the backward processes the states in inverted order).
   */
  private var stateIndex: Int = 0

  /**
   * The params errors accumulator of the transform vectors.
   */
  private var transformErrorsAccumulator = ParamsErrorsAccumulator<FeedforwardLayerParameters>()

  /**
   * The params errors accumulator of the recurrent context network.
   */
  private var recurrentErrorsAccumulator = ParamsErrorsAccumulator<NetworkParameters>()

  /**
   * The params errors accumulator of the attention structure
   */
  private var attentionErrorsAccumulator = ParamsErrorsAccumulator<AttentionParameters>()

  /**
   * The errors of the recurrent context, set at each backward step.
   */
  private lateinit var recurrentErrors: DenseNDArray

  /**
   * Perform the back-propagation from the output errors.
   *
   * @param outputErrors the errors to propagate
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    this.initBackward()

    (0 until outputErrors.size).reversed().forEach { stateIndex ->

      this.stateIndex = stateIndex

      this.backwardStep(
        outputErrors = outputErrors[stateIndex],
        isFirstState = stateIndex == 0,
        isLastState = stateIndex == outputErrors.lastIndex)
    }

    // The errors in the 'contextErrorsAccumulator' are already averaged thanks to the recurrent processor
    this.recurrentErrorsAccumulator.accumulate(this.network.recurrentProcessor.getParamsErrors(copy = false))
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of the [network]
   */
  fun getParamsErrors(copy: Boolean = true) = PointerNetworkParameters(
    recurrentParams = this.recurrentErrorsAccumulator.getParamsErrors(copy = copy),
    transformParams = this.transformErrorsAccumulator.getParamsErrors(copy = copy),
    attentionParams = this.attentionErrorsAccumulator.getParamsErrors(copy = copy))

  /**
   * Initialize the structures used during a backward.
   */
  private fun initBackward() {

    this.recurrentErrorsAccumulator.reset()
  }

  /**
   * A single step of backward.
   *
   * @param outputErrors the errors of a single output array
   * @param isFirstState a boolean indicating if this is the first state of the sequence (the last of the backward)
   * @param isLastState a boolean indicating if this is the last state of the sequence (the first of the backward)
   */
  private fun backwardStep(outputErrors: DenseNDArray, isFirstState: Boolean, isLastState: Boolean) {

    // TODO()
  }
}
