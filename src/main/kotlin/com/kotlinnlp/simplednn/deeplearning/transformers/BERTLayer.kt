/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.deeplearning.attention.multihead.MultiHeadAttentionNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A single BERT layer.
 *
 * @property params the layer parameters
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property useDropout whether to apply the attention dropout during the [forward]
 * @property id a unique ID
 */
internal class BERTLayer(
  val params: BERTParameters,
  override val propagateToInput: Boolean = false,
  override val useDropout: Boolean = false,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * The errors accumulator.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator()

  /**
   * The input sequence.
   */
  private lateinit var inputSequence: List<AugmentedArray<DenseNDArray>>

  /**
   * The multi-head scaled-dot attention network.
   */
  private val multiHeadAttention = MultiHeadAttentionNetwork(
    model = this.params.attention,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput)

  /**
   * The multi-head norm batch processor.
   */
  private val multiHeadNorm: BatchFeedforwardProcessor<DenseNDArray> =
    BatchFeedforwardProcessor(model = this.params.multiHeadNorm, propagateToInput = true, useDropout = false)

  /**
   * The batch of output feed-forward processors.
   */
  private val outputFF: BatchFeedforwardProcessor<DenseNDArray> =
    BatchFeedforwardProcessor(model = this.params.outputFF, propagateToInput = true, useDropout = false)

  /**
   * The output norm batch processor.
   */
  private val outputNorm: BatchFeedforwardProcessor<DenseNDArray> =
    BatchFeedforwardProcessor(model = this.params.outputNorm, propagateToInput = true, useDropout = false)

  /**
   * @param input the input sequence
   *
   * @return the encoded sequence
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.inputSequence = input.map { AugmentedArray(it) }

    return this.forwardOutput(this.forwardAttention(input))
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.errorsAccumulator.clear()

    val inputErrors: List<DenseNDArray> = this.backwardAttention(this.backwardOutput(outputErrors))

    if (this.propagateToInput)
      this.inputSequence.zip(inputErrors).forEach { (input, errors) -> input.assignErrors(errors) }

    this.errorsAccumulator.averageErrors()
  }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.errorsAccumulator.getParamsErrors(copy = copy)

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> =
    this.inputSequence.map { if (copy) it.errors.copy() else it.errors }

  /**
   * The attention component forward.
   *
   * @param input the input sequence
   *
   * @return the output arrays of the attention
   */
  private fun forwardAttention(input: List<DenseNDArray>): List<DenseNDArray> {

    val attentionArrays: List<DenseNDArray> = this.multiHeadAttention.forward(input)

    return this.multiHeadNorm.forward(input.zip(attentionArrays).map { it.first.sum(it.second) })
  }

  /**
   * The output component forward.
   *
   * @param attentionArrays the attention arrays
   *
   * @return the output arrays
   */
  private fun forwardOutput(attentionArrays: List<DenseNDArray>): List<DenseNDArray> {

    val ffOutputs: List<DenseNDArray> = this.outputFF.forward(attentionArrays)

    return this.outputNorm.forward(attentionArrays.zip(ffOutputs).map { it.first.sum(it.second) })
  }

  /**
   * The output component backward.
   *
   * @param outputErrors the output errors
   *
   * @return the errors of the attention arrays
   */
  private fun backwardOutput(outputErrors: List<DenseNDArray>): List<DenseNDArray> {

    val normErrors: List<DenseNDArray> = this.backwardNorm(errors = outputErrors, processor = this.outputNorm)

    this.outputFF.backward(normErrors)
    this.errorsAccumulator.accumulate(this.outputFF.getParamsErrors(copy = false), copy = false)

    return this.outputFF.getInputErrors(copy = false).zip(normErrors).map { (inputErrors, errors) ->
      inputErrors.sum(errors)
    }
  }

  /**
   * The attention component backward.
   *
   * @param attentionErrors the errors of the attention arrays
   *
   * @return the layer input errors
   */
  private fun backwardAttention(attentionErrors: List<DenseNDArray>): List<DenseNDArray> {

    val normErrors: List<DenseNDArray> = this.backwardNorm(errors = attentionErrors, processor = this.multiHeadNorm)

    this.multiHeadAttention.backward(normErrors)
    this.errorsAccumulator.accumulate(this.multiHeadAttention.getParamsErrors(copy = false), copy = false)

    return if (this.propagateToInput)
      this.multiHeadAttention.getInputErrors().zip(attentionErrors).map { it.first.assignSum(it.second) }
    else
      listOf()
  }

  /**
   * Execute the backward of the given norm processor.
   *
   * @param errors the output errors
   * @param processor the norm processor on which to call the backward
   *
   * @return the inputs errors
   */
  private fun backwardNorm(errors: List<DenseNDArray>,
                           processor: BatchFeedforwardProcessor<DenseNDArray>): List<DenseNDArray> =
    processor.let {
      it.backward(errors)
      this.errorsAccumulator.accumulate(it.getParamsErrors(copy = false))
      it.getInputErrors(copy = false)
    }
}
