/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A Bidirectional Encoder Representations from Transformers.
 *
 * @property model the BERT model
 * @param fineTuning whether to train the last layer only (default false)
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property useDropout whether to apply the attention dropout during the [forward]
 * @property id a unique ID
 */
class BERT(
  val model: BERTModel,
  fineTuning: Boolean = false,
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
   * The pre-calculated positional encodings.
   */
  private val positionalEncodings: MutableList<DenseNDArray> = mutableListOf()

  /**
   * The input arrays scale factor.
   */
  private val inputScale: Double = sqrt(this.model.inputSize.toDouble())

  /**
   * The BERT layers.
   */
  private val layers: List<BERTLayer> = this.model.layers.mapIndexed { i, params ->
    BERTLayer(
      params = params,
      propagateToInput = i > 0 || this.propagateToInput,
      useDropout = this.useDropout)
  }

  /**
   * The trainable layers.
   * Only the last of the stack in case of model fine tuning.
   */
  private val trainableLayers: List<BERTLayer> = if (fineTuning) this.layers.takeLast(1) else this.layers

  /**
   * @param input the input sequence
   *
   * @return the encoded sequence
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    var sequence: List<DenseNDArray> = this.addPositionalEncodings(input.map { it.prod(this.inputScale) })

    this.layers.forEach {
      sequence = it.forward(sequence)
    }

    return sequence
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.errorsAccumulator.clear()

    var errors: List<DenseNDArray> = outputErrors

    this.trainableLayers.reversed().forEach {

      it.backward(errors)

      this.errorsAccumulator.accumulate(it.getParamsErrors(copy = false), copy = false)

      errors = it.getInputErrors(copy = false)
    }

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
    this.layers.first().getInputErrors(copy = false).map { it.prod(this.inputScale) }

  /**
   * Add positional encodings to the input sequence, in-place.
   *
   * @param inputSequence the input sequence
   *
   * @return the input sequence with the positional encodings added in-place
   */
  private fun addPositionalEncodings(inputSequence: List<DenseNDArray>): List<DenseNDArray> =
    inputSequence.mapIndexed { pos, array -> array.assignSum(this.getPositionalEncoding(pos)) }

  /**
   * @param pos the position of an input array within the sequence
   *
   * @return the positioanl encoding for the given position
   */
  private fun getPositionalEncoding(pos: Int): DenseNDArray =
    this.positionalEncodings.getOrElse(pos) { this.buildPositionalEncoding() }

  /**
   * Build a new positional encoding and add it to the pre-calculated arrays.
   *
   * @return a new positional encoding for the (last + 1) position
   */
  private fun buildPositionalEncoding(): DenseNDArray {

    val pos: Int = this.positionalEncodings.size
    val encoding: DenseNDArray = DenseNDArrayFactory.arrayOf(
      DoubleArray(
        size = this.model.inputSize,
        init = { i ->
          if (i % 2 == 0)
            sin(pos / 10000.0.pow(i.toDouble() / this.model.inputSize))
          else
            cos(pos / 10000.0.pow(i.toDouble() / this.model.inputSize))
        }
      )
    )

    this.positionalEncodings.add(encoding)

    return encoding
  }
}
