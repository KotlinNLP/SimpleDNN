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
   * A pool of Feedforward Layers used to build the attention arrays of the Attention Network.
   */
  val transformLayersPool: FeedforwardLayersPool<DenseNDArray> =
    FeedforwardLayersPool(
      inputType = LayerType.Input.Dense,
      activationFunction = Tanh(),
      params = this.model.transformParams)

  /**
   * A pool of Attention Networks.
   */
  val attentionNetworksPool: AttentionNetworksPool<DenseNDArray> =
    AttentionNetworksPool(
      model = this.model.attentionParams,
      inputType = LayerType.Input.Dense)

  /**
   * The processor for the recurrent context network.
   */
  val contextProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(this.model.recurrentContextNetwork)

  /**
   * The pool of Feedforward Neural Processors used to interpolate the output of the Attention Network together with the
   * recurrent context.
   */
  val outputNetworkPool: FeedforwardNeuralProcessorsPool<DenseNDArray> =
    FeedforwardNeuralProcessorsPool(this.model.outputNetwork)

  /**
   * The zeros array used as null state encoding.
   */
  val initialStateEncoding = DenseNDArrayFactory.zeros(Shape(this.model.recurrentContextSize))

  /**
   * The list of transform layers groups used during the last forward.
   */
  val usedTransformLayers = mutableListOf<List<FeedforwardLayerStructure<DenseNDArray>>>()

  /**
   * The list of Attention Networks used during the last forward.
   */
  val usedAttentionNetworks = mutableListOf<AttentionNetwork<DenseNDArray>>()

  /**
   * The output processors used during the last forward.
   */
  val usedOutputProcessors = mutableListOf<FeedforwardNeuralProcessor<DenseNDArray>>()

  /**
   * The size of the processing sequence (set at the first forward state).
   */
  var sequenceSize: Int = 0
    private set

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
   * @param firstState a boolean indicating if this is the first state
   * @param sequence the sequence
   * @param lastPrediction the dense representation of the last prediction
   *
   * @return the output array of the network
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
   * Back-propagation of the errors.
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
