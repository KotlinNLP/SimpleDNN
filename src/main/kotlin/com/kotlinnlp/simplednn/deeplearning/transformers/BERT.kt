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
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * A Bidirectional Encoder Representations from Transformers.
 *
 * @property model the BERT model
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property useDropout whether to apply the attention dropout during the [forward]
 * @property id a unique ID
 */
class BERT(
  val model: BERTModel,
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
   * One list per stacked layer.
   */
  private lateinit var attentionLayers: List<List<ScaledDotAttentionLayer>>

  /**
   * The pools of concat layers of the multi-head attention outputs, one per stacked layer.
   */
  private val multiHeadMergePools: List<ConcatFFLayersPool<DenseNDArray>> = this.model.layers.map {
    ConcatFFLayersPool<DenseNDArray>(params = it.multiHeadMerge, inputType = LayerType.Input.Dense)
  }

  /**
   * The concat layers of the multi-head attention outputs that have been used for the last forward.
   * One list per stacked layer.
   */
  private lateinit var multiHeadConcatLayers: List<List<ConcatFFLayer<DenseNDArray>>>

  /**
   * The pools of output feed-forward networks, one per stacked layer.
   */
  private val outputFFPools: List<FeedforwardNeuralProcessorsPool<DenseNDArray>> = this.model.layers.map {
    FeedforwardNeuralProcessorsPool<DenseNDArray>(model = it.outputFF, propagateToInput = true, useDropout = false)
  }

  /**
   * The output feed-forward networks that have been used for the last forward.
   * One list per stacked layer.
   */
  private lateinit var outputFFNetworks: List<List<FeedforwardNeuralProcessor<DenseNDArray>>>

  /**
   * The error of the norm scalar parameters accumulated during the last backward, one per stacked layer.
   */
  private val normScalarsErrors: MutableList<Double> = MutableList(size = this.model.layers.size, init = { 0.0 })

  /**
   * @param input the input sequence
   *
   * @return the encoded sequence
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.setInputSequence(input)

    var sequence: List<DenseNDArray> = this.inputSequence.map { it.values }

    this.model.layers.indices.forEach { i ->
      sequence = this.forwardOutput(this.forwardAttention(sequence, layerIndex = i), layerIndex = i)
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
    this.normScalarsErrors.indices.forEach { this.normScalarsErrors[it] = 0.0 }

    val inputErrors: List<DenseNDArray> = this.backwardLayers(outputErrors)

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
   * Set the input sequence combining it with the positional encoding.
   *
   * @param inputSequence the list of arrays of input
   */
  private fun setInputSequence(inputSequence: List<DenseNDArray>) {

    this.inputSequence = this.addPositionalEncodings(inputSequence.map { AugmentedArray(it) })

    this.attentionLayers = this.model.layers.map { params ->
      params.attention.map { attentionParams ->
        ScaledDotAttentionLayer(
          inputArrays = List(size = this.inputSequence.size, init = { AugmentedArray.zeros(this.model.inputSize) }),
          params = attentionParams,
          inputDropout = if (this.useDropout) this.model.dropout else 0.0)
      }
    }

    this.multiHeadMergePools.forEach { it.releaseAll() }
    this.multiHeadConcatLayers = this.multiHeadMergePools.map { pool -> inputSequence.indices.map { pool.getItem() } }

    this.outputFFPools.forEach { it.releaseAll() }
    this.outputFFNetworks = this.outputFFPools.map { pool -> inputSequence.indices.map { pool.getItem() } }
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
          array.values[i + 1] += cos(pos / 10000.0.pow((2.0 * i + 1.0) / this.model.inputSize))
      }
    }

    return inputSequence
  }

  /**
   * The attention component forward of a given stacked layer.
   *
   * @param inputs the input sequence
   * @param layerIndex the stacked layer index
   *
   * @return the output arrays of the attention
   */
  private fun forwardAttention(inputs: List<DenseNDArray>, layerIndex: Int): List<DenseNDArray> {

    val attentionLayers: List<ScaledDotAttentionLayer> = this.attentionLayers[layerIndex]
    val concatLayers: List<ConcatFFLayer<DenseNDArray>> = this.multiHeadConcatLayers[layerIndex]
    val normScalar: Double = this.model.layers[layerIndex].normScalar

    attentionLayers.forEach {
      it.inputArrays.zip(inputs).forEach { (array, input) -> array.assignValues(input) }
      it.forward(useDropout = this.useDropout)
    }

    return concatLayers.mapIndexed { i, concatLayer ->

      concatLayer.inputArrays.zip(attentionLayers).forEach { (mergeInput, attentionLayer) ->
        mergeInput.assignValues(attentionLayer.outputArrays[i].values)
      }
      concatLayer.forward()

      this.inputSequence[i].values.sum(concatLayer.outputArray.values.prod(normScalar))
    }
  }

  /**
   * The output component forward of a given stacked layer.
   *
   * @param attentionArrays the attention arrays
   * @param layerIndex the stacked layer index
   *
   * @return the output arrays
   */
  private fun forwardOutput(attentionArrays: List<DenseNDArray>, layerIndex: Int): List<DenseNDArray> {

    val outputFF: List<FeedforwardNeuralProcessor<DenseNDArray>> = this.outputFFNetworks[layerIndex]
    val normScalar: Double = this.model.layers[layerIndex].normScalar

    return outputFF.zip(attentionArrays).map { (outputFF, attentionArray) ->

      outputFF.forward(attentionArray)

      attentionArray.sum(outputFF.getOutput(copy = false).prod(normScalar))
    }
  }

  /**
   * Backward of all the BERT layers, starting from the last.
   *
   * @param outputErrors the output errors
   *
   * @return the input errors (empty if [propagateToInput] is false)
   */
  private fun backwardLayers(outputErrors: List<DenseNDArray>): List<DenseNDArray> {

    var errors: List<DenseNDArray> = outputErrors

    this.model.layers.indices.reversed().forEach { i ->
      errors = this.backwardAttention(this.backwardOutput(errors, layerIndex = i), layerIndex = i)
    }

    this.model.layers.zip(this.normScalarsErrors).forEach { (layer, scalarError) ->
      this.errorsAccumulator.accumulate(
        layer.normScalarParam.buildDenseErrors(DenseNDArrayFactory.arrayOf(doubleArrayOf(scalarError))))
    }

    return errors
  }

  /**
   * The output component backward of a given stacked layer.
   *
   * @param outputErrors the output errors
   * @param layerIndex the stacked layer index
   *
   * @return the errors of the attention arrays
   */
  private fun backwardOutput(outputErrors: List<DenseNDArray>, layerIndex: Int): List<DenseNDArray> {

    val outputFF: List<FeedforwardNeuralProcessor<DenseNDArray>> = this.outputFFNetworks[layerIndex]
    val normScalar: Double = this.model.layers[layerIndex].normScalar

    return outputFF.zip(outputErrors).map { (outputFF, errors) ->

      outputFF.backward(errors.prod(normScalar))
      this.errorsAccumulator.accumulate(outputFF.getParamsErrors(copy = false))

      this.normScalarsErrors[layerIndex] += errors.prod(outputFF.getOutput(copy = false)).sum()

      errors.sum(outputFF.getInputErrors(copy = false))
    }
  }

  /**
   * The attention component backward of a given stacked layer.
   *
   * @param attentionErrors the errors of the attention arrays
   * @param layerIndex the stacked layer index
   *
   * @return the layer input errors
   */
  private fun backwardAttention(attentionErrors: List<DenseNDArray>, layerIndex: Int): List<DenseNDArray> {

    val attentionLayers: List<ScaledDotAttentionLayer> = this.attentionLayers[layerIndex]
    val concatLayers: List<ConcatFFLayer<DenseNDArray>> = this.multiHeadConcatLayers[layerIndex]
    val normScalar: Double = this.model.layers[layerIndex].normScalar
    val propagateToInput: Boolean = layerIndex > 0 || this.propagateToInput

    attentionErrors.forEachIndexed { i, errors ->

      val concatLayer: ConcatFFLayer<DenseNDArray> = concatLayers[i]

      concatLayer.setErrors(errors.prod(normScalar))
      this.errorsAccumulator.accumulate(concatLayer.backward(propagateToInput = true))

      this.normScalarsErrors[layerIndex] += errors.prod(concatLayer.outputArray.values).sum()

      attentionLayers.zip(concatLayer.getInputErrors(copy = false)).forEach { (attentionLayer, concatErrors) ->
        attentionLayer.outputArrays[i].assignErrors(concatErrors)
      }
    }

    attentionLayers.forEach { this.errorsAccumulator.accumulate(it.backward(propagateToInput)) }

    return if (propagateToInput)
      this.backwardAttentionInput(attentionLayers = attentionLayers, attentionErrors = attentionErrors)
    else
      listOf()
  }

  /**
   * Errors back-propagation to the input of the attention layers of a given stacked layer.
   *
   * @param attentionLayers the multi-head attention layers of a given stacked layer
   * @param attentionErrors the output errors of the given multi-head attention
   *
   * @return the input errors of the given multi-head attention
   */
  private fun backwardAttentionInput(
    attentionLayers: List<ScaledDotAttentionLayer>,
    attentionErrors: List<DenseNDArray>
  ): List<DenseNDArray> {

    val inputErrors: List<DenseNDArray> = attentionErrors.map { it.copy() }

    attentionLayers.forEach {
      inputErrors.zip(it.inputArrays).forEach { (errors, input) -> errors.assignSum(input.errors) }
    }

    return inputErrors
  }
}
