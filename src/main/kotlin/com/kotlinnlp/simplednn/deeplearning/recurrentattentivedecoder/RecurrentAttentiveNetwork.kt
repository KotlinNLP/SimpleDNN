/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.FeedforwardLayersPool
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [RecurrentAttentiveNetwork].
 *
 * It encodes an input sequence of N arrays into a parallel output sequence of N arrays, giving the focus to the i-th
 * array at the i-th encoding step thanks to a recurrent context network.
 *
 * At each step the current state is encoded combining the whole input sequence with the recurrent context (that is
 * the memory of this system). Then the encoded state is forwarded to the output through the output network together
 * with the recurrent context.
 *
 * The recurrent context is encoded using an RNN that receives the last encoded state and a context label vector as
 * input.
 *
 * @property model the model of the network
 */
class RecurrentAttentiveNetwork(val model: RecurrentAttentiveNetworkModel) {

  /**
   * The size of the currently processing sequence (set at the first forward state).
   */
  var sequenceSize: Int = 0
    private set

  /**
   * A boolean that indicates that the network is being trained.
   */
  var trainingMode: Boolean = false
    private set

  /**
   * A pool of Feedforward Layers used to build the attention arrays of the Attention Network.
   */
  val transformLayersPool: FeedforwardLayersPool<DenseNDArray> =
    FeedforwardLayersPool(
      inputType = LayerType.Input.Dense,
      activationFunction = Tanh(),
      params = this.model.transformParams)

  /**
   * A pool of Attention Networks used to encode the current state.
   */
  val attentionNetworksPool: AttentionNetworksPool<DenseNDArray> =
    AttentionNetworksPool(
      model = this.model.attentionParams,
      inputType = LayerType.Input.Dense)

  /**
   * The processor of the recurrent context network.
   */
  val recurrentContextProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(this.model.recurrentContextNetwork)

  /**
   * The pool of Feedforward Neural Processors used to interpolate the encoded state together with the recurrent
   * context.
   */
  val outputNetworkPool: FeedforwardNeuralProcessorsPool<DenseNDArray> =
    FeedforwardNeuralProcessorsPool(this.model.outputNetwork)

  /**
   * The list of transform layers groups used during the last forward.
   */
  val usedTransformLayers = mutableListOf<List<FeedforwardLayerStructure<DenseNDArray>>>()

  /**
   * The list of Attention Networks used to encode all the states of the current input sequence.
   */
  val usedAttentionNetworks = mutableListOf<AttentionNetwork<DenseNDArray>>()

  /**
   * The output processors used during the last forward.
   */
  val usedOutputProcessors = mutableListOf<FeedforwardNeuralProcessor<DenseNDArray>>()

  /**
   * The forward helper.
   */
  private val forwardHelper = ForwardHelper(network = this)

  /**
   * The backward helper.
   */
  private val backwardHelper = BackwardHelper(network = this)

  /**
   * Forward.
   *
   * @param inputSequence the input sequence
   * @param lastPredictionLabel the context label vector used to encode the memory of the last prediction
   * @param firstState a boolean indicating if this is the first state
   * @param trainingMode a boolean to enable when the network is being trained (necessary to call a backward later)
   *
   * @return the output array of the network
   */
  fun forward(inputSequence: List<DenseNDArray>,
              lastPredictionLabel: DenseNDArray?,
              firstState: Boolean,
              trainingMode: Boolean): DenseNDArray {

    if (firstState) this.sequenceSize = inputSequence.size

    this.trainingMode = trainingMode

    return this.forwardHelper.forward(
      firstState = firstState,
      inputSequence = inputSequence,
      lastPredictionLabel = lastPredictionLabel)
  }

  /**
   * Back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    require(this.trainingMode) { "Cannot call a backward if the network has not being forwarded in training mode" }

    this.backwardHelper.backward(outputErrors = outputErrors)
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of this network
   */
  fun getParamsErrors(copy: Boolean = true): RecurrentAttentiveNetworkParameters =
    this.backwardHelper.getParamsErrors(copy = copy)

  /**
   * @return the errors of the sequence
   */
  fun getInputSequenceErrors(): List<DenseNDArray> = this.backwardHelper.inputSequenceErrors

  /**
   * @return the errors of the context label vectors
   */
  fun getContextLabelsErrors(): List<DenseNDArray> = this.backwardHelper.contextLabelsErrors
}
