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
   * The structure used to store the params errors of the transform layers during the backward.
   */
  private lateinit var _transformLayerParamsErrors: FeedforwardLayerParameters

  /**
   * The structure used to store the params errors of the Attention Network during the backward.
   */
  private lateinit var _attentionNetworkParamsErrors: AttentionNetworkParameters

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
    params = this.network.model.contextNetwork.model,
    updateMethod = this.network.updateMethod)

  /**
   *
   */
  private var stepIndex: Int = 0

  /**
   * The recurrent errors of the context.
   */
  var contextErrors: DenseNDArray? = null

  /**
   * The error of the processing sequence.
   */
  var sequenceErrors: Array<DenseNDArray>? = null

  /**
   * The error of the predictions.
   */
  val predictionsErrors = mutableListOf<DenseNDArray>()

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

    this.sequenceErrors = this.initSequenceErrors()
    this.predictionsErrors.clear()

    var recurrentAttentionErrors: DenseNDArray? = null

    (0 until outputErrors.size).reversed().forEach { stepIndex ->

      this.stepIndex = stepIndex // important

      val (attentionPart, contextPart) = this.getAttentionAndContextOutputErrors(
        outputErrors = outputErrors[stepIndex])

      // update the context errors and the sequence errors
      this.backwardAttentionAndTransform(
        outputErrors = if (stepIndex == outputErrors.lastIndex)
          attentionPart
        else
          attentionPart.assignSum(recurrentAttentionErrors!!))

      if (stepIndex > 0) {
        this.contextErrors!!.assignSum(contextPart)

        val (prevAttentionErrors, prevPredictionErrors) = this.contextBackwardStep()

        this.predictionsErrors.add(prevPredictionErrors)

        recurrentAttentionErrors = prevAttentionErrors
      }
    }

    this.contextNetworkOptimizer.accumulate(this.network.contextProcessor.getParamsErrors(copy = true))
  }

  /**
   * @return an array of DenseNDArray initialized at zero with the same size of the current sequence
   */
  private fun initSequenceErrors() = Array(
    size = this.network.sequenceSize,
    init = { DenseNDArrayFactory.zeros(Shape(this.network.model.inputSize)) })

  /**
   * @param outputErrors the output errors of the output network
   *
   * @return the partitions of the errors between the Attention and the Context
   */
  private fun getAttentionAndContextOutputErrors(outputErrors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val inputErrors = this.getOutputNetworkInputErrors(outputErrors = outputErrors)

    val splitErrors: Array<DenseNDArray> = inputErrors.splitV(
      this.network.model.attentionParams.outputSize,
      this.network.model.contextSize
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

    val processor = this.network.usedOutputProcessors[this.stepIndex]

    processor.backward(outputErrors, propagateToInput = true)
    this.outputNetworkOptimizer.accumulate(processor.getParamsErrors(copy = true))

    return processor.getInputErrors(copy = true)
  }

  /**
   * Backward of the Attention Network used during the given decoding [stepIndex].
   *
   * @param outputErrors the output errors of the given Attention Network
   *
   * @return the context-errors
   */
  private fun backwardAttentionAndTransform(outputErrors: DenseNDArray) {

    val transformLayers = this.network.usedTransformLayers[this.stepIndex]
    val (inputErrors, attentionErrors) = this.backwardAttention(outputErrors = outputErrors)

    this.resetContextErrors()

    attentionErrors.forEachIndexed { itemIndex, errors ->

      val transformErrors: DenseNDArray = this.backwardTransformLayer(
        layer = transformLayers[itemIndex],
        outputErrors = errors)

      val (sequencePart, contextPart) = this.splitTransformErrors(errors = transformErrors)

      this.contextErrors!!.assignSum(contextPart)
      this.sequenceErrors!![itemIndex].assignSum(sequencePart.sum(inputErrors[itemIndex]))
    }
  }

  /**
   * Set the [contextErrors] to zeros.
   */
  fun resetContextErrors() {
    this.contextErrors = DenseNDArrayFactory.zeros(Shape(this.network.model.contextSize))
  }

  /**
   * Backward of the Attention Network.
   *
   * @param outputErrors the output errors
   *
   * @return a Pair of InputErrors, AttentionErrors
   */
  private fun backwardAttention(outputErrors: DenseNDArray): Pair<Array<DenseNDArray>, Array<DenseNDArray>> {

    val attentionNetwork = this.network.usedAttentionNetworks[this.stepIndex]
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

    return layer.inputArray.errors.copy()
  }

  /**
   * A single backward step of the recurrent network.
   */
  private fun contextBackwardStep(): Pair<DenseNDArray, DenseNDArray> {

    this.network.contextProcessor.backwardStep(
      outputErrors = this.contextErrors!!,
      propagateToInput = true)

    return this.splitContextInputErrors(this.network.contextProcessor.getInputErrors(
      elementIndex = this.stepIndex - 1, // important
      copy = true))
  }

  /**
   * @param errors the errors of an encoded attention array of the Attention Network
   *
   * @return a Pair containing partitions of the input item and context errors
   */
  private fun splitTransformErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.network.model.inputSize,
      this.network.model.contextSize)

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
      this.network.model.labelSize)

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * @return the transform layers params errors
   */
  private fun getTransformParamsErrors(): FeedforwardLayerParameters = try {
      this._transformLayerParamsErrors
    } catch (e: UninitializedPropertyAccessException) {
      this._transformLayerParamsErrors = this.network.usedTransformLayers.last().last().params.copy() as FeedforwardLayerParameters
      this._transformLayerParamsErrors
    }

  /**
   * @return the Attention Network params errors
   */
  private fun getAttentionParamsErrors(): AttentionNetworkParameters = try {
      this._attentionNetworkParamsErrors
    } catch (e: UninitializedPropertyAccessException) {
      this._attentionNetworkParamsErrors = this.network.usedAttentionNetworks.last().model.copy()
      this._attentionNetworkParamsErrors
    }
}