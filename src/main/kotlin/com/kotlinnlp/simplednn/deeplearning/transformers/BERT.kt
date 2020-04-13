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
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayersPool
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayersPool
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayersPool
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
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
   * The pool of sum layers for the concatenated multi-head attention outputs.
   */
  private val multiHeadSumPool = SumLayersPool<DenseNDArray>(params = this.model.sum, inputType = LayerType.Input.Dense)

  /**
   * The sum layers for the merged multi-head attention outputs that have been used for the last forward.
   */
  private lateinit var multiHeadSumLayers: List<SumLayer<DenseNDArray>>

  /**
   * The pool of output feed-forward layers.
   */
  private val outputFFPool = FeedforwardLayersPool<DenseNDArray>(
    params = this.model.outputFF,
    inputType = LayerType.Input.Dense,
    activationFunction = null)

  /**
   * The output feed-forward layers that have been used for the last forward.
   */
  private lateinit var outputFFLayers: List<FeedforwardLayer<DenseNDArray>>

  /**
   * The pool of sum layers for the feed-forward outputs.
   */
  private val outputSumPool = SumLayersPool<DenseNDArray>(params = this.model.sum, inputType = LayerType.Input.Dense)

  /**
   * The sum layers for the feed-forward outputs that have been used for the last forward.
   */
  private lateinit var outputSumLayers: List<SumLayer<DenseNDArray>>

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

    this.attentionLayers = List(
      size = this.model.multiHeadStack,
      init = {
        ScaledDotAttentionLayer(
          inputArrays = if (this.propagateToInput) this.inputSequence.map { it.clone() } else this.inputSequence,
          params = this.model.attention,
          inputDropout = if (this.useDropout) this.model.dropout else 0.0)
      }
    )

    this.multiHeadMergePool.releaseAll()
    this.multiHeadConcatLayers = inputSequence.indices.map { this.multiHeadMergePool.getItem() }

    this.multiHeadSumPool.releaseAll()
    this.multiHeadSumLayers = inputSequence.indices.map { this.multiHeadSumPool.getItem() }

    this.outputFFPool.releaseAll()
    this.outputFFLayers = inputSequence.indices.map { this.outputFFPool.getItem() }

    this.outputSumPool.releaseAll()
    this.outputSumLayers = inputSequence.indices.map { this.outputSumPool.getItem() }
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

    return this.multiHeadConcatLayers.zip(this.multiHeadSumLayers).mapIndexed { i, (concatLayer, sumLayer) ->

      concatLayer.inputArrays.zip(this.attentionLayers).forEach { (mergeInput, attentionLayer) ->
        mergeInput.assignValues(attentionLayer.outputArrays[i].values)
      }
      concatLayer.forward()

      sumLayer.inputArrays[0].assignValues(this.inputSequence[i].values)
      sumLayer.inputArrays[1].assignValues(concatLayer.outputArray.values.prod(this.model.normScalar))
      sumLayer.forward()

      sumLayer.outputArray.values
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

    this.inputSequence.indices.map { i ->

      val outputLayer: FeedforwardLayer<DenseNDArray> = this.outputFFLayers[i]
      val attentionArray: DenseNDArray = attentionArrays[i]
      val sumLayer: SumLayer<DenseNDArray> = this.outputSumLayers[i]

      outputLayer.apply { setInput(attentionArray); forward() }

      sumLayer.inputArrays[0].assignValues(attentionArray)
      sumLayer.inputArrays[1].assignValues(outputLayer.outputArray.values.prod(this.model.normScalar))
      sumLayer.forward()

      sumLayer.outputArray.values
    }

  /**
   * The output component backward.
   *
   * @param outputErrors the output errors
   *
   * @return the errors of the attention arrays
   */
  private fun backwardOutput(outputErrors: List<DenseNDArray>): List<DenseNDArray> =

    outputErrors.mapIndexed { i, errors ->

      val outputLayer: FeedforwardLayer<DenseNDArray> = this.outputFFLayers[i]
      val sumLayer: SumLayer<DenseNDArray> = this.outputSumLayers[i]

      sumLayer.setErrors(errors)
      this.errorsAccumulator.accumulate(sumLayer.backward(propagateToInput = true))

      val sumErrors: List<DenseNDArray> = sumLayer.getInputErrors(copy = false)
      outputLayer.setErrors(sumErrors[1].prod(this.model.normScalar))
      this.errorsAccumulator.accumulate(outputLayer.backward(propagateToInput = true))

      this.normScalarError += sumErrors[1].prod(outputLayer.outputArray.values).sum()

      sumErrors[0].sum(outputLayer.inputArray.errors)
    }

  /**
   * The attention component backward.
   *
   * @param attentionErrors the errors of the attention arrays
   */
  private fun backwardAttention(attentionErrors: List<DenseNDArray>) {

    attentionErrors.mapIndexed { i, errors ->

      val concatLayer: ConcatFFLayer<DenseNDArray> = this.multiHeadConcatLayers[i]
      val sumLayer: SumLayer<DenseNDArray> = this.multiHeadSumLayers[i]

      sumLayer.setErrors(errors)
      this.errorsAccumulator.accumulate(sumLayer.backward(propagateToInput = true))

      val sumErrors: List<DenseNDArray> = sumLayer.getInputErrors(copy = false)
      concatLayer.setErrors(sumErrors[1].prod(this.model.normScalar))
      this.errorsAccumulator.accumulate(concatLayer.backward(propagateToInput = true))

      this.normScalarError += sumErrors[1].prod(concatLayer.outputArray.values).sum()

      this.attentionLayers.zip(concatLayer.getInputErrors(copy = false)).forEach { (attentionLayer, concatErrors) ->
        attentionLayer.outputArrays[i].assignErrors(concatErrors)
      }

      if (this.propagateToInput)
        this.inputSequence[i].assignErrors(sumErrors[0])
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
