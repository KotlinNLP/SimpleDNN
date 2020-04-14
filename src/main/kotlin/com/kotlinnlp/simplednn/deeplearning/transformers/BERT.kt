/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayersPool
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.math.pow
import kotlin.math.sin

/**
 * A Bidirectional Encoder Representations from Transformers.
 *
 * @property model the parameters of the model of the network
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property useDropout whether to apply the attention dropout during the [forward]
 * @property id a unique ID
 */
class BERT(
  val model: BERTParameters,
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
   * The scaled-dot attention layers used for the last forward.
   */
  private lateinit var attentionLayers: List<ScaledDotAttentionLayer>

  /**
   * The pool of concat layers of the multi-head attention outputs.
   */
  private val multiHeadMergePool = ConcatFFLayersPool<DenseNDArray>(
    params = this.model.multiHeadMerge,
    inputType = LayerType.Input.Dense)

  /**
   * The concat layers of the multi-head attention outputs that have been used for the last forward.
   */
  private lateinit var multiHeadConcatLayers: List<ConcatFFLayer<DenseNDArray>>

  /**
   * The pool of output feed-forward networks.
   */
  private val outputFFPool = FeedforwardNeuralProcessorsPool<DenseNDArray>(
    model = this.model.outputFF,
    propagateToInput = true,
    useDropout = false)

  /**
   * The output feed-forward networks that have been used for the last forward.
   */
  private lateinit var outputFFNetworks: List<FeedforwardNeuralProcessor<DenseNDArray>>

  /**
   * The error of the norm scalar parameter accumulated during the last backward.
   */
  private var normScalarError: Double = 0.0

  /**
   * @param input the input sequence
   *
   * @return the encoded sequence
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.setInputSequence(input)

    return this.forwardOutput(attentionArrays = this.forwardAttention())
  }

  /**
   * Propagate the output errors using the gradient descent algorithm.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.errorsAccumulator.clear()
    this.normScalarError = 0.0

    this.backwardAttention(attentionErrors = this.backwardOutput(outputErrors))

    if (this.propagateToInput)
      this.backwardInput()

    this.errorsAccumulator.accumulate(
      this.model.normScalarParam.buildDenseErrors(DenseNDArrayFactory.arrayOf(doubleArrayOf(this.normScalarError))))

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
   * Set the input sequence combining it with the positional encoding.
   *
   * @param inputSequence the list of arrays of input
   */
  private fun setInputSequence(inputSequence: List<DenseNDArray>) {

    this.inputSequence = this.addPositionalEncodings(inputSequence.map { AugmentedArray(it) })

    this.attentionLayers = this.model.attention.map { params ->
      ScaledDotAttentionLayer(
        inputArrays = if (this.propagateToInput) this.inputSequence.map { it.clone() } else this.inputSequence,
        params = params,
        inputDropout = if (this.useDropout) this.model.dropout else 0.0)
    }

    this.multiHeadMergePool.releaseAll()
    this.multiHeadConcatLayers = inputSequence.indices.map { this.multiHeadMergePool.getItem() }

    this.outputFFPool.releaseAll()
    this.outputFFNetworks = inputSequence.indices.map { this.outputFFPool.getItem() }
  }

  /**
   * Add positional encodings to the input sequence.
   *
   * @param inputSequence the input sequence
   *
   * @return the input sequence with the positional encodings added in-place
   */
  private fun addPositionalEncodings(
    inputSequence: List<AugmentedArray<DenseNDArray>>
  ): List<AugmentedArray<DenseNDArray>> {

    inputSequence.forEachIndexed { pos, array ->

      (0 .. array.values.length / 2).forEach { i ->

        array.values[i] += sin(pos / 10000.0.pow(2.0 * i / this.model.inputSize))

        if ((i + 1) < array.values.length)
          array.values[i + 1] += sin(pos / 10000.0.pow((2.0 * i + 1.0) / this.model.inputSize))
      }
    }

    return inputSequence
  }

  /**
   * The attention component forward.
   *
   * @return the output arrays of the attention
   */
  private fun forwardAttention(): List<DenseNDArray> {

    this.attentionLayers.forEach { it.forward(useDropout = this.useDropout) }

    return this.multiHeadConcatLayers.mapIndexed { i, concatLayer ->

      concatLayer.inputArrays.zip(this.attentionLayers).forEach { (mergeInput, attentionLayer) ->
        mergeInput.assignValues(attentionLayer.outputArrays[i].values)
      }
      concatLayer.forward()

      this.inputSequence[i].values.sum(concatLayer.outputArray.values.prod(this.model.normScalar))
    }
  }

  /**
   * The output component forward.
   *
   * @param attentionArrays the attention arrays
   *
   * @return the output arrays
   */
  private fun forwardOutput(attentionArrays: List<DenseNDArray>): List<DenseNDArray> =

    this.outputFFNetworks.zip(attentionArrays).map { (outputFF, attentionArray) ->

      outputFF.forward(attentionArray)

      attentionArray.sum(outputFF.getOutput(copy = false).prod(this.model.normScalar))
    }

  /**
   * The output component backward.
   *
   * @param outputErrors the output errors
   *
   * @return the errors of the attention arrays
   */
  private fun backwardOutput(outputErrors: List<DenseNDArray>): List<DenseNDArray> =

    this.outputFFNetworks.zip(outputErrors).map { (outputFF, errors) ->

      outputFF.backward(errors.prod(this.model.normScalar))
      this.errorsAccumulator.accumulate(outputFF.getParamsErrors(copy = false))

      this.normScalarError += errors.prod(outputFF.getOutput(copy = false)).sum()

      errors.sum(outputFF.getInputErrors(copy = false))
    }

  /**
   * The attention component backward.
   *
   * @param attentionErrors the errors of the attention arrays
   */
  private fun backwardAttention(attentionErrors: List<DenseNDArray>) {

    attentionErrors.forEachIndexed { i, errors ->

      val concatLayer: ConcatFFLayer<DenseNDArray> = this.multiHeadConcatLayers[i]

      concatLayer.setErrors(errors.prod(this.model.normScalar))
      this.errorsAccumulator.accumulate(concatLayer.backward(propagateToInput = true))

      this.normScalarError += errors.prod(concatLayer.outputArray.values).sum()

      this.attentionLayers.zip(concatLayer.getInputErrors(copy = false)).forEach { (attentionLayer, concatErrors) ->
        attentionLayer.outputArrays[i].assignErrors(concatErrors)
      }

      if (this.propagateToInput)
        this.inputSequence[i].assignErrors(errors)
    }

    this.attentionLayers.forEach {
      this.errorsAccumulator.accumulate(it.backward(propagateToInput = this.propagateToInput))
    }
  }

  /**
   * Back-propagation to the input.
   */
  private fun backwardInput() {

    val inputs: Sequence<AugmentedArray<DenseNDArray>> = this.inputSequence.asSequence()

    this.attentionLayers.forEach { attentionLayer ->

      inputs.zip(attentionLayer.inputArrays.asSequence()).forEach { (input, attentionInput) ->
        input.errors.assignSum(attentionInput.errors)
      }
    }
  }
}
