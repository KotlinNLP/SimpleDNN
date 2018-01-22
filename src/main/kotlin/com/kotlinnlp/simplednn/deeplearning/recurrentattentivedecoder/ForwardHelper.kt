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
   * @param inputSequence the input sequence
   * @param lastPredictionLabel the context label vector used to encode the memory of the last prediction
   * @param firstState a boolean indicating if this is the first state
   */
  fun forward(inputSequence: List<DenseNDArray>,
              lastPredictionLabel: DenseNDArray?,
              firstState: Boolean): DenseNDArray {

    if (firstState) this.resetHistory()

    val recurrentContext: DenseNDArray = if (firstState)
      this.network.initialStateEncoding
    else
      this.forwardRecurrentContext(firstState = firstState, lastPredictionLabel = lastPredictionLabel)

    val stateEncoding: DenseNDArray = this.encodeState(sequence = inputSequence, context = recurrentContext)

    return this.getOutputProcessor().forward(concatVectorsV(stateEncoding, recurrentContext))
  }

  /**
   * Reset the recurrent history of the network.
   */
  private fun resetHistory() {

    this.network.transformLayersPool.releaseAll()
    this.network.usedTransformLayers.clear()

    this.network.stateEncodersPool.releaseAll()
    this.network.usedStateEncoders.clear()

    this.network.outputNetworkPool.releaseAll()
    this.network.usedOutputProcessors.clear()
  }

  /***
   * @param lastPredictionLabel the context label vector used to encode the memory of the last prediction
   * @param firstState a boolean indicating if this is first state
   *
   * @return the recurrent context for the current state
   */
  private fun forwardRecurrentContext(lastPredictionLabel: DenseNDArray?, firstState: Boolean): DenseNDArray =
    this.network.recurrentContextProcessor.forward(
      featuresArray = concatVectorsV(this.getLastEncodedState(), lastPredictionLabel!!),
      firstState = firstState)

  /**
   * Encode the current state.
   *
   * @param sequence the sequence to decode
   * @param context the recurrent context
   *
   * @return the encoded state as result of the [AttentionNetwork]
   */
  private fun encodeState(sequence: List<DenseNDArray>, context: DenseNDArray): DenseNDArray {

    val attentionNetwork = this.getAttentionNetwork()

    return attentionNetwork.forward(
      inputSequence = ArrayList(sequence.map { AugmentedArray(values = it) }),
      attentionSequence = this.buildAttentionSequence(sequence = sequence, context = context))
  }

  /**
   * @param sequence the input sequence
   * @param context the recurrent context
   *
   * @return the sequence of attention arrays
   */
  private fun buildAttentionSequence(sequence: List<DenseNDArray>,
                                     context: DenseNDArray): ArrayList<DenseNDArray> {

    val transformLayers = this.getTransformLayers(size = sequence.size)

    return ArrayList(transformLayers.zip(sequence).map { (layer, inputArray) ->

      layer.setInput(concatVectorsV(inputArray, context))
      layer.forward()

      layer.outputArray.values
    })
  }

  /**
   * Get an available transform layer.
   *
   * @param size the number of transform layer to build
   *
   * @return an available transform layer
   */
  private fun getTransformLayers(size: Int): List<FeedforwardLayerStructure<DenseNDArray>> {

    val layers = List(size = size, init = { this.network.transformLayersPool.getItem() })

    // TODO: use always the first of the pool during training
    this.network.usedTransformLayers.add(layers)

    return layers
  }

  /**
   * Get an available Attention Network.
   *
   * @return an available Attention Network
   */
  private fun getAttentionNetwork(): AttentionNetwork<DenseNDArray> {

    val attentionNetwork = this.network.stateEncodersPool.getItem()

    // TODO: use always the first of the pool during training
    this.network.usedStateEncoders.add(attentionNetwork)

    return attentionNetwork
  }

  /**
   * Get an available output processor.
   *
   * @return an available output processor
   */
  private fun getOutputProcessor(): FeedforwardNeuralProcessor<DenseNDArray> {

    val processor = this.network.outputNetworkPool.getItem()

    // TODO: use always the first of the pool during training
    this.network.usedOutputProcessors.add(processor)

    return processor
  }

  /**
   * @return the last encoded state
   */
  private fun getLastEncodedState(): DenseNDArray = this.network.usedStateEncoders.last().getOutput()
}
