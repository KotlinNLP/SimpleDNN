/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessorsPool
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ScheduledUpdater
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.FeedforwardLayersPool
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The [RecurrentAttentiveNetwork].
 */
class RecurrentAttentiveNetwork(
  val model: RecurrentAttentiveNetworkModel,
  val updateMethod: UpdateMethod<*>) : ScheduledUpdater {

  /**
   * The pool of layers used to create the attention arrays of the Attention Network.
   */
  val attentionNetworksPool: AttentionNetworksPool<DenseNDArray> =
    AttentionNetworksPool(
      model = this.model.attentionParams,
      inputType = LayerType.Input.Dense)

  /**
   * The pool attention networks used to encode the state.
   */
  val transformLayersPool: FeedforwardLayersPool<DenseNDArray> =
    FeedforwardLayersPool(
      inputType = LayerType.Input.Dense,
      activationFunction = Tanh(),
      params = this.model.transformParams)

  /**
   * The pool of FeedforwardNeuralProcessors layers used to encode the output.
   */
  val outputNetworkPool: FeedforwardNeuralProcessorsPool<DenseNDArray> =
    FeedforwardNeuralProcessorsPool(this.model.outputNetwork)

  /**
   * The processor for the recurrent network
   */
  val contextProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(this.model.contextNetwork)

  /**
   * The zeros array used as null memory encoding.
   */
  val initialEncodedState = DenseNDArrayFactory.zeros(Shape(this.model.contextSize))

  /**
   * The list of active Attention Networks
   */
  val usedAttentionNetworks = mutableListOf<AttentionNetwork<DenseNDArray>>()

  /**
   * The features layers used during the last forward.
   * (Its usage makes the training no thread safe).
   */
  val usedOutputProcessors = mutableListOf<FeedforwardNeuralProcessor<DenseNDArray>>()

  /**
   * The list of transform layers groups used during the last forward.
   * (Its usage makes the training no thread safe).
   */
  val usedTransformLayers = mutableListOf<List<FeedforwardLayerStructure<DenseNDArray>>>()

  /**
   * The size of the processing sequence (updated at the first forward state)
   */
  var sequenceSize: Int = 0

  /**
   * The forward helper.
   */
  private val forwardHelper = ForwardHelper(network = this)

  /**
   * The backward helper.
   */
  private val backwardHelper = BackwardHelper(network = this)

  /**
   * @param firstState a boolean indicating if this is the first state
   * @param sequence the sequence
   * @param lastPrediction the dense representation of the last prediction
   */
  fun forward(firstState: Boolean,
              sequence: List<DenseNDArray>,
              lastPrediction: DenseNDArray?): DenseNDArray {

    if (firstState) this.sequenceSize = sequence.size

    return this.forwardHelper.forward(
      firstState = firstState,
      sequence = sequence,
      lastPrediction = lastPrediction)
  }

  /**
   * Start the back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: List<DenseNDArray>)
    = this.backwardHelper.backward(outputErrors = outputErrors)

  /**
   * @return the errors of the sequence
   */
  fun getItemsErrors(): Array<DenseNDArray> = this.backwardHelper.sequenceErrors!!

  /**
   * @return the errors of the predictions
   */
  fun getPredictionsErrors(): List<DenseNDArray> = this.backwardHelper.predictionsErrors

  /**
   * Update the parameters of the neural elements associated to this [ScheduledUpdater].
   */
  override fun update() {
    this.backwardHelper.update()
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.backwardHelper.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.backwardHelper.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.backwardHelper.newExample()
  }
}
