/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * A Bidirectional Encoder Representations from Transformers.
 *
 * @property model the BERT model
 * @param fineTuning whether to train the last layer only (default false)
 * @param masksEnabled whether to consider the token [BERTModel.FuncToken.MASK] as a functional token if present in a
 *                     sentence (default = false)
 * @property propagateToInput whether to propagate the errors to the input word embeddings during the [backward]
 * @property id a unique ID
 */
class BERT(
  val model: BERTModel,
  fineTuning: Boolean = false,
  private val masksEnabled: Boolean = false,
  override val propagateToInput: Boolean = false,
  override val id: Int = 0
) : NeuralProcessor<
  List<String>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  NeuralProcessor.NoInputErrors // InputErrorsType
  > {

  /**
   * Dropout not available.
   */
  override val useDropout: Boolean = false

  /**
   * The errors accumulator.
   */
  private val errorsAccumulator = ParamsErrorsAccumulator()

  /**
   * The BERT layers.
   */
  private val layers: List<BERTLayer> = this.model.layers.mapIndexed { i, params ->
    BERTLayer(params = params, propagateToInput = i > 0 || this.propagateToInput)
  }

  /**
   * The trainable layers.
   * Only the last of the stack in case of model fine tuning.
   */
  private val trainableLayers: List<BERTLayer> = if (fineTuning) this.layers.takeLast(1) else this.layers

  /**
   * The norm layer batch processor.
   */
  private val embNorm: BatchFeedforwardProcessor<DenseNDArray> =
    BatchFeedforwardProcessor(model = this.model.embNorm, propagateToInput = true, useDropout = false)

  /**
   * The input sequence as pairs of <form, encoding>.
   */
  private lateinit var inputSequence: List<Pair<String, DenseNDArray>>

  /**
   * The errors associated to the padding terms.
   */
  private val zeroErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.inputSize))

  /**
   * @param input the input sequence
   *
   * @return the encoded sequence
   */
  override fun forward(input: List<String>): List<DenseNDArray> {

    var encodings: List<DenseNDArray> = this.embNorm.forward(this.encodeSequence(input))

    this.layers.forEach {
      encodings = it.forward(encodings)
    }

    return encodings.subList(1, encodings.lastIndex) // remove the padding tokens
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.errorsAccumulator.clear()

    var errors: List<DenseNDArray> = listOf(this.zeroErrors) + outputErrors + listOf(this.zeroErrors)

    this.trainableLayers.reversed().forEach {

      it.backward(errors)

      this.errorsAccumulator.accumulate(it.getParamsErrors(copy = false), copy = false)

      errors = it.getInputErrors(copy = false)
    }

    this.backwardInput(errors)

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
   * Input errors not provided.
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * Encode the input sequence adding the padding functional tokens and associating the embeddings properly.
   * The tokens with their encodings are saved into [inputSequence].
   *
   * @param input the input sequence
   *
   * @return the input encodings
   */
  private fun encodeSequence(input: List<String>): List<DenseNDArray> {

    fun encodeToken(token: String, pos: Int, isFunc: Boolean = false): Pair<String, DenseNDArray> {

      val wordEmb: ParamsArray = if (isFunc || (this.masksEnabled && token == BERTModel.FuncToken.MASK.form))
        this.model.funcEmb[BERTModel.FuncToken.byForm(token)]
      else
        this.model.wordEmb!![token]

      val encoding: DenseNDArray = wordEmb.values
        .sum(this.getPositionalEncoding(pos))
        .assignSum(this.model.tokenTypeEmb[0].values)

      return token to encoding
    }

    // -------------------------------

    this.inputSequence = listOf(encodeToken(token = BERTModel.FuncToken.CLS.form, pos = 0, isFunc = true)) +
      input.mapIndexed { i, it -> encodeToken(token = it, pos = i + 1) } +
      listOf(encodeToken(token = BERTModel.FuncToken.SEP.form, pos = input.lastIndex + 2, isFunc = true))

    return this.inputSequence.unzip().second
  }

  /**
   * @param pos the position of an input array within the sequence
   *
   * @return the positional encoding for the given position
   */
  private fun getPositionalEncoding(pos: Int): DenseNDArray =
    this.model.positionalEmb
      .getOrSet(pos) { ParamsArray(this.buildPositionalEncoding(pos)) }
      .values

  /**
   * @param pos the ordinal position
   *
   * @return a new positional encoding for the given position
   */
  private fun buildPositionalEncoding(pos: Int): DenseNDArray = DenseNDArrayFactory.arrayOf(
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

  /**
   * Execute the backward of the input normalization and embeddings.
   *
   * @param errors the input normalization errors
   */
  private fun backwardInput(errors: List<DenseNDArray>) {

    val embErrors: List<DenseNDArray> = this.embNorm.let {
      it.backward(errors)
      this.errorsAccumulator.accumulate(it.getParamsErrors(copy = false))
      it.getInputErrors(copy = false)
    }

    this.inputSequence.zip(embErrors).forEachIndexed { i, (input, inputErrors) ->

      val isFuncToken: Boolean = i == 0 || i == this.inputSequence.lastIndex
      val token: String = input.first

      val wordEmb: ParamsArray = if (isFuncToken)
        this.model.funcEmb[BERTModel.FuncToken.byForm(token)]
      else
        this.model.wordEmb!![token]

      if (isFuncToken || this.propagateToInput)
        this.errorsAccumulator.accumulate(wordEmb, inputErrors)

      this.errorsAccumulator.accumulate(this.model.positionalEmb[i], inputErrors)
      this.errorsAccumulator.accumulate(this.model.tokenTypeEmb[0], inputErrors)
    }
  }
}
