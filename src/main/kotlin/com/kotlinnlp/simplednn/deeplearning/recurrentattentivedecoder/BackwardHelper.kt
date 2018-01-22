/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.ScheduledUpdater
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The backward helper of the [RecurrentAttentiveNetwork].
 */
class BackwardHelper(private val network: RecurrentAttentiveNetwork) : ScheduledUpdater {

  /**
   * The error of the predictions.
   */
  val predictionsErrors = mutableListOf<DenseNDArray>()

  /**
   * The error of the processing sequence.
   */
  lateinit var sequenceErrors: Array<DenseNDArray>
    private set

  /**
   * The index of the current state (the backward process the states in inverted order).
   */
  private var stateIndex: Int = 0

  /**
   * The recurrent errors of the context.
   */
  private lateinit var contextErrors: DenseNDArray

  /**
   * The structure used to store the params errors of the transform layers during the backward.
   */
  private lateinit var transformLayerParamsErrors: FeedforwardLayerParameters

  /**
   * The structure used to store the params errors of the Attention Network during the backward.
   */
  private lateinit var attentionNetworkParamsErrors: AttentionNetworkParameters

  /**
   * The optimizer of the transform layers.
   */
  private val transformLayerOptimizer: ParamsOptimizer<FeedforwardLayerParameters> = ParamsOptimizer(
    params = this.network.model.transformParams,
    updateMethod = this.network.updateMethod)

  /**
   * The optimizer of the attention network.
   */
  private val attentionNetworkOptimizer: ParamsOptimizer<AttentionNetworkParameters> = ParamsOptimizer(
    params = this.network.model.attentionParams,
    updateMethod = this.network.updateMethod)

  /**
   * The optimizer of the final feed-forward network.
   */
  private val outputNetworkOptimizer: ParamsOptimizer<NetworkParameters> = ParamsOptimizer(
    params = this.network.model.outputNetwork.model,
    updateMethod = this.network.updateMethod)

  /**
   * The optimizer of the recurrent network.
   */
  private val contextNetworkOptimizer: ParamsOptimizer<NetworkParameters> = ParamsOptimizer(
    params = this.network.model.recurrentContextNetwork.model,
    updateMethod = this.network.updateMethod)

  /**
   * Update the parameters.
   */
  override fun update() {
    this.outputNetworkOptimizer.update()
    this.transformLayerOptimizer.update()
    this.attentionNetworkOptimizer.update()
    this.contextNetworkOptimizer.update()
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {

    if (this.network.updateMethod is EpochScheduling) {
      this.network.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {

    if (this.network.updateMethod is BatchScheduling) {
      this.network.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {

    if (this.network.updateMethod is ExampleScheduling) {
      this.network.updateMethod.newExample()
    }
  }

  /**
   * Perform the back-propagation from the output errors.
   *
   * @param outputErrors the errors to propagate
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    this.initSequenceErrors()
    this.predictionsErrors.clear()

    var recurrentAttentionErrors: DenseNDArray? = null

    (0 until outputErrors.size).reversed().forEach { stateIndex ->

      this.stateIndex = stateIndex // important

      val (attentionPart, contextPart) = this.getAttentionAndContextOutputErrors(
        outputErrors = outputErrors[stateIndex])

      // update the context errors and the sequence errors
      this.backwardAttentionAndTransform(
        outputErrors = if (stateIndex == outputErrors.lastIndex)
          attentionPart
        else
          attentionPart.assignSum(recurrentAttentionErrors!!))

      if (stateIndex > 0) {
        this.contextErrors.assignSum(contextPart)

        val (prevAttentionErrors, prevPredictionErrors) = this.contextBackwardStep()

        this.predictionsErrors.add(prevPredictionErrors)

        recurrentAttentionErrors = prevAttentionErrors
      }
    }

    this.contextNetworkOptimizer.accumulate(this.network.recurrentContextProcessor.getParamsErrors(copy = false))
  }

  /**
   * Initialize the [sequenceErrors] with arrays of zeros (an amount equal to the size of the current input sequence).
   */
  private fun initSequenceErrors() {
    this.sequenceErrors = Array(
      size = this.network.sequenceSize,
      init = { DenseNDArrayFactory.zeros(Shape(this.network.model.inputSize)) })
  }

  /**
   * @param outputErrors the output errors of the output network
   *
   * @return the partitions of the errors between the Attention and the Context
   */
  private fun getAttentionAndContextOutputErrors(outputErrors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val inputErrors = this.getOutputNetworkInputErrors(outputErrors = outputErrors)

    val splitErrors: Array<DenseNDArray> = inputErrors.splitV(
      this.network.model.attentionParams.outputSize,
      this.network.model.recurrentContextSize
    )

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * Perform a backward of the output network.
   *
   * @param outputErrors the input errors of the output network
   *
   * @return the input errors of the output network
   */
  private fun getOutputNetworkInputErrors(outputErrors: DenseNDArray): DenseNDArray {

    val processor = this.network.usedOutputProcessors[this.stateIndex]

    processor.backward(outputErrors, propagateToInput = true)
    this.outputNetworkOptimizer.accumulate(processor.getParamsErrors(copy = false))

    return processor.getInputErrors(copy = false)
  }

  /**
   * Backward of the Attention Network used during the given decoding [stateIndex].
   *
   * @param outputErrors the output errors of the given Attention Network
   *
   * @return the context-errors
   */
  private fun backwardAttentionAndTransform(outputErrors: DenseNDArray) {

    val transformLayers = this.network.usedTransformLayers[this.stateIndex]
    val (inputErrors, attentionErrors) = this.backwardAttention(outputErrors = outputErrors)

    this.resetContextErrors()

    attentionErrors.forEachIndexed { itemIndex, errors ->

      val transformErrors: DenseNDArray = this.backwardTransformLayer(
        layer = transformLayers[itemIndex],
        outputErrors = errors)

      val (sequencePart, contextPart) = this.splitTransformErrors(errors = transformErrors)

      this.contextErrors.assignSum(contextPart)
      this.sequenceErrors[itemIndex].assignSum(sequencePart.sum(inputErrors[itemIndex]))
    }
  }

  /**
   * Set the [contextErrors] to zeros.
   */
  fun resetContextErrors() {
    this.contextErrors = DenseNDArrayFactory.zeros(Shape(this.network.model.recurrentContextSize))
  }

  /**
   * Backward of the Attention Network.
   *
   * @param outputErrors the output errors
   *
   * @return a Pair of InputErrors, AttentionErrors
   */
  private fun backwardAttention(outputErrors: DenseNDArray): Pair<Array<DenseNDArray>, Array<DenseNDArray>> {

    val attentionNetwork = this.network.usedStateEncoders[this.stateIndex]
    val paramsErrors = this.getAttentionParamsErrors()

    attentionNetwork.backward(
      outputErrors = outputErrors,
      paramsErrors = paramsErrors,
      propagateToInput = true)

    this.attentionNetworkOptimizer.accumulate(paramsErrors)

    return Pair(attentionNetwork.getInputErrors(), attentionNetwork.getAttentionErrors())
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

    this.transformLayerOptimizer.accumulate(paramsErrors)

    return layer.inputArray.errors
  }

  /**
   * A single backward step of the recurrent network.
   */
  private fun contextBackwardStep(): Pair<DenseNDArray, DenseNDArray> {

    this.network.recurrentContextProcessor.backwardStep(
      outputErrors = this.contextErrors,
      propagateToInput = true)

    return this.splitContextInputErrors(this.network.recurrentContextProcessor.getInputErrors(
      elementIndex = this.stateIndex - 1, // important
      copy = false))
  }

  /**
   * @param errors the errors of an encoded attention array of the Attention Network
   *
   * @return a Pair containing partitions of the input item and context errors
   */
  private fun splitTransformErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.network.model.inputSize,
      this.network.model.recurrentContextSize)

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * @param errors the input errors of a memory encoding
   *
   * @return a Pair (attentionErrors, predictionErrors)
   */
  private fun splitContextInputErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

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
    this.attentionNetworkParamsErrors = this.network.usedStateEncoders.last().model.copy()
    this.attentionNetworkParamsErrors
  }
}
