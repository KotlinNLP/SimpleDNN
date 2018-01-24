/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentiverecurrentnetwork

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The backward helper of the [AttentiveRecurrentNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class BackwardHelper(private val network: AttentiveRecurrentNetwork) {

  /**
   * The error of the context label vectors (the first is always null).
   */
  val contextLabelsErrors = mutableListOf<DenseNDArray?>()

  /**
   * The list of errors of the current input sequence.
   */
  lateinit var inputSequenceErrors: List<DenseNDArray>
    private set

  /**
   * The index of the current state (the backward processes the states in inverted order).
   */
  private var stateIndex: Int = 0

  /**
   * The params errors accumulator of the transform layer.
   */
  private var transformLayerAccumulator = ParamsErrorsAccumulator<FeedforwardLayerParameters>()

  /**
   * The params errors accumulator of the attention network.
   */
  private var attentionNetworkAccumulator = ParamsErrorsAccumulator<AttentionNetworkParameters>()

  /**
   * The params errors accumulator of the recurrent context network.
   */
  private var contextErrorsAccumulator = ParamsErrorsAccumulator<NetworkParameters>()

  /**
   * The params errors accumulator of the output network.
   */
  private var outputErrorsAccumulator = ParamsErrorsAccumulator<NetworkParameters>()

  /**
   * The structure used to store the params errors of the transform layers during the backward.
   */
  private lateinit var transformLayerParamsErrors: FeedforwardLayerParameters

  /**
   * The structure used to store the params errors of the Attention Network during the backward.
   */
  private lateinit var attentionNetworkParamsErrors: AttentionNetworkParameters

  /**
   * The errors of the recurrent context, set at each backward step.
   */
  private lateinit var recurrentContextErrors: DenseNDArray

  /**
   * The recurrent errors of the state encoding.
   */
  private lateinit var recurrentStateEncodingErrors: DenseNDArray

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
    this.contextErrorsAccumulator.accumulate(this.network.recurrentContextProcessor.getParamsErrors(copy = false))

    this.transformLayerAccumulator.averageErrors()
    this.attentionNetworkAccumulator.averageErrors()
    this.outputErrorsAccumulator.averageErrors()
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of the [network]
   */
  fun getParamsErrors(copy: Boolean = true) = AttentiveRecurrentNetworkParameters(
    transformParams = this.transformLayerAccumulator.getParamsErrors(copy = copy),
    attentionParams = this.attentionNetworkAccumulator.getParamsErrors(copy = copy),
    recurrentContextParams = this.contextErrorsAccumulator.getParamsErrors(copy = copy),
    outputParams = this.outputErrorsAccumulator.getParamsErrors(copy = copy))

  /**
   * Initialize the structures used during a backward.
   */
  private fun initBackward() {

    this.initSequenceErrors()

    this.contextLabelsErrors.clear()

    this.transformLayerAccumulator.reset()
    this.attentionNetworkAccumulator.reset()
    this.contextErrorsAccumulator.reset()
    this.outputErrorsAccumulator.reset()
  }

  /**
   * Initialize the [inputSequenceErrors] with arrays of zeros (an amount equal to the size of the current input
   * sequence).
   */
  private fun initSequenceErrors() {
    this.inputSequenceErrors = List(
      size = this.network.sequenceSize,
      init = { DenseNDArrayFactory.zeros(Shape(this.network.model.inputSize)) })
  }

  /**
   * A single step of backward.
   *
   * @param outputErrors the errors of a single output array
   * @param isFirstState a boolean indicating if this is the first state of the sequence (the last of the backward)
   * @param isLastState a boolean indicating if this is the last state of the sequence (the first of the backward)
   */
  private fun backwardStep(outputErrors: DenseNDArray, isFirstState: Boolean, isLastState: Boolean) {

    val (stateEncodingPart, recurrentContextPart) = this.splitOutputNetworkErrors(
      outputNetworkErrors = this.getOutputNetworkInputErrors(outputErrors)
    )

    val stateEncoderInputErrors = this.backwardStateEncoder(
      stateEncodingErrors = if (isLastState)
        stateEncodingPart
      else
        stateEncodingPart.assignSum(this.recurrentStateEncodingErrors))

    this.propagateStateEncodingErrors(stateEncoderInputErrors)

    this.recurrentContextErrors.assignSum(recurrentContextPart)

    if (isFirstState) {

      this.contextLabelsErrors.add(0, null)

    } else {

      val (prevStateEncodingErrors, contextLabelErrors) = this.recurrentContextBackwardStep()

      this.contextLabelsErrors.add(0, contextLabelErrors)
      this.recurrentStateEncodingErrors = prevStateEncodingErrors
    }
  }

  /**
   * @param outputNetworkErrors the input errors of the output network
   *
   * @return the partitions of the errors between the state encoding and the recurrent context
   */
  private fun splitOutputNetworkErrors(outputNetworkErrors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = outputNetworkErrors.splitV(
      this.network.model.attentionParams.outputSize,
      this.network.model.recurrentContextSize
    )

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * Perform a backward of the output network.
   *
   * @param outputErrors the errors of the output at the [stateIndex] state
   *
   * @return the input errors of the output network
   */
  private fun getOutputNetworkInputErrors(outputErrors: DenseNDArray): DenseNDArray {

    val processor = this.network.usedOutputProcessors[this.stateIndex]

    processor.backward(outputErrors, propagateToInput = true)
    this.outputErrorsAccumulator.accumulate(processor.getParamsErrors(copy = false))

    return processor.getInputErrors(copy = false)
  }

  /**
   * Backward of the Attention Network used to encode the state at the [stateIndex] state.
   *
   * @param stateEncodingErrors the errors of the state encoding at the [stateIndex] state
   *
   * @return a list of Pairs of input arrays errors and attention arrays errors
   */
  private fun backwardStateEncoder(stateEncodingErrors: DenseNDArray): List<Pair<DenseNDArray, DenseNDArray>> {

    val attentionNetwork = this.network.usedAttentionNetworks[this.stateIndex]
    val paramsErrors = this.getAttentionParamsErrors()

    attentionNetwork.backward(outputErrors = stateEncodingErrors, paramsErrors = paramsErrors, propagateToInput = true)

    this.attentionNetworkAccumulator.accumulate(paramsErrors)

    return attentionNetwork.getInputErrors().zip(attentionNetwork.getAttentionErrors())
  }

  /**
   *
   * @param stateEncoderInputErrors the input errors of the state encoder used at the [stateIndex] state
   */
  private fun propagateStateEncodingErrors(stateEncoderInputErrors: List<Pair<DenseNDArray, DenseNDArray>>) {

    val transformLayers = this.network.usedTransformLayers[this.stateIndex]

    stateEncoderInputErrors.forEachIndexed { itemIndex, (inputArrayErrors, attentionArrayErrors) ->

      val transformErrors: DenseNDArray = this.backwardTransformLayer(
        layer = transformLayers[itemIndex],
        outputErrors = attentionArrayErrors)

      val (inputArrayTransformPart, focusContextPart) = this.splitTransformErrors(errors = transformErrors)

      this.recurrentContextErrors = if (itemIndex == 0)
        focusContextPart
      else
        this.recurrentContextErrors.assignSum(focusContextPart)

      this.inputSequenceErrors[itemIndex].assignSum(inputArrayTransformPart.sum(inputArrayErrors))
    }
  }

  /**
   * A single transform layer backward.
   *
   * @param layer a transform layer
   * @param outputErrors the errors of the output
   *
   * @return the errors of the input
   */
  private fun backwardTransformLayer(layer: FeedforwardLayerStructure<DenseNDArray>,
                                     outputErrors: DenseNDArray): DenseNDArray {

    val paramsErrors = this.getTransformParamsErrors()

    layer.setErrors(outputErrors)
    layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

    this.transformLayerAccumulator.accumulate(paramsErrors)

    return layer.inputArray.errors
  }

  /**
   * A single backward step of the recurrent context network.
   *
   * @return the partitions of the errors between the previous state encoding and the context label
   */
  private fun recurrentContextBackwardStep(): Pair<DenseNDArray, DenseNDArray> {

    this.network.recurrentContextProcessor.backwardStep(
      outputErrors = this.recurrentContextErrors,
      propagateToInput = true)

    return this.splitRNNInputErrors(errors = this.network.recurrentContextProcessor.getInputErrors(
      elementIndex = this.stateIndex - 1, // in the i-th state the encoding of the previous state (i-1) is used as input
      copy = false))
  }

  /**
   * @param errors the errors of an encoded attention array of the Attention Network
   *
   * @return the partitions of the errors between the input array and the recurrent context used as focus
   */
  private fun splitTransformErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.network.model.inputSize,
      this.network.model.recurrentContextSize)

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * @param errors the input errors of a recurrent context encoding
   *
   * @return the partitions of the errors between the state encoding and the context label
   */
  private fun splitRNNInputErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.network.model.attentionParams.outputSize,
      this.network.model.contextLabelSize)

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * @return the transform layers params errors
   */
  private fun getTransformParamsErrors(): FeedforwardLayerParameters = try {
    this.transformLayerParamsErrors
  } catch (e: UninitializedPropertyAccessException) {
    this.transformLayerParamsErrors =
      this.network.usedTransformLayers.last().last().params.copy() as FeedforwardLayerParameters
    this.transformLayerParamsErrors
  }

  /**
   * @return the Attention Network params errors
   */
  private fun getAttentionParamsErrors(): AttentionNetworkParameters = try {
    this.attentionNetworkParamsErrors
  } catch (e: UninitializedPropertyAccessException) {
    this.attentionNetworkParamsErrors = this.network.usedAttentionNetworks.last().model.copy()
    this.attentionNetworkParamsErrors
  }
}
