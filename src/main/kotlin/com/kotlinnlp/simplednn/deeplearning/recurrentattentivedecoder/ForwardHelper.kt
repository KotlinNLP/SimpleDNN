/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The forward helper of the [RecurrentAttentiveNetwork].
 */
class ForwardHelper(private val network: RecurrentAttentiveNetwork) {

  /**
   * @param firstState a boolean indicating if this is the first state
   * @param inputSequence the input sequence
   * @param lastPredictionLabel the context label vector used to encode the memory of the last prediction
   */
  fun forward(firstState: Boolean,
              inputSequence: List<DenseNDArray>,
              lastPredictionLabel: DenseNDArray?): DenseNDArray {

    val processor = this.getFeedforwardProcessor(firstState = firstState)

    val context: DenseNDArray = this.forwardContext(
      firstState = firstState,
      lastPrediction = lastPredictionLabel)

    val attention: DenseNDArray = this.forwardAttention(
      firstState = firstState,
      sequence = inputSequence,
      context = context)

    return processor.forward(concatVectorsV(attention, context))
  }

  /***
   * @param firstState a boolean indicating if this is first state
   * @param lastPrediction the dense representation of the last prediction
   *
   * @return the recurrent context for the current state
   */
  private fun forwardContext(firstState: Boolean, lastPrediction: DenseNDArray?): DenseNDArray =
    if (firstState)
      this.network.initialStateEncoding
    else
      this.network.recurrentContextProcessor.forward(
        featuresArray = concatVectorsV(this.getLastAttention(), lastPrediction!!),
        firstState = firstState)

  /**
   * @param firstState a boolean indicating if this is first state
   * @param sequence the sequence to decode
   * @param context the recurrent context
   *
   * @return the result of the [AttentionNetwork]
   */
  private fun forwardAttention(firstState: Boolean,
                               sequence: List<DenseNDArray>,
                               context: DenseNDArray): DenseNDArray {

    val attentionNetwork = this.getAttentionNetwork(firstState = firstState)

    return attentionNetwork.forward(
      inputSequence = ArrayList(sequence.map { AugmentedArray(values = it) }),
      attentionSequence = this.buildAttentionSequence(
        firstState = firstState,
        sequence = sequence,
        context = context))
  }

  /**
   * @param firstState a boolean indicating if this is the first state
   * @param sequence the sequence to ecnode
   * @param context the recurrent context
   *
   * @return the attention sequence
   */
  private fun buildAttentionSequence(firstState: Boolean,
                                     sequence: List<DenseNDArray>,
                                     context: DenseNDArray): ArrayList<DenseNDArray> {

    val transformLayers = this.getTransformLayers(firstState = firstState, size = sequence.size)

    return ArrayList(transformLayers.zip(sequence).map { (layer, item) ->

      layer.setInput(concatVectorsV(item, context))
      layer.forward()

      layer.outputArray.values.copy()
    })
  }

  /**
   * Get an available transform layer.
   *
   * @param firstState a boolean indicating if this is the first state
   *
   * @return an available transform layer
   */
  private fun getTransformLayers(firstState: Boolean, size: Int): List<FeedforwardLayerStructure<DenseNDArray>> {

    if (firstState) {
      this.network.transformLayersPool.releaseAll()
      this.network.usedTransformLayers.clear()
    }

    val layers = List(size = size, init = { this.network.transformLayersPool.getItem() })

    this.network.usedTransformLayers.add(layers)

    return layers
  }

  /**
   * Get an available feed-forward processor.
   *
   * @param firstState a boolean indicating if this is the first state
   *
   * @return an available feed-forward layer
   */
  private fun getFeedforwardProcessor(firstState: Boolean): FeedforwardNeuralProcessor<DenseNDArray> {

    if (firstState) {
      this.network.usedOutputProcessors.clear()
      this.network.outputNetworkPool.releaseAll()
    }

    val processor = this.network.outputNetworkPool.getItem()

    // TODO: use always the first of the pool during training
    this.network.usedOutputProcessors.add(processor)

    return processor
  }

  /**
   * Get an available Attention Network.
   *
   * @param firstState a boolean indicating if this is the first state
   *
   * @return an available Attention Network
   */
  private fun getAttentionNetwork(firstState: Boolean): AttentionNetwork<DenseNDArray> {

    if (firstState) {
      this.network.usedStateEncoders.clear()
      this.network.stateEncodersPool.releaseAll()
    }

    val attentionNetwork = this.network.stateEncodersPool.getItem()

    // TODO: use always the first of the pool during training
    this.network.usedStateEncoders.add(attentionNetwork)

    return attentionNetwork
  }

  /**
   * @return the output values of the last attention
   */
  private fun getLastAttention(): DenseNDArray = this.network.usedStateEncoders.last().getOutput(copy = true)
}
