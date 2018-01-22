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
 *
 * @property model the model of the network
 * @property updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
 */
class RecurrentAttentiveNetwork(
  val model: RecurrentAttentiveNetworkModel,
  val updateMethod: UpdateMethod<*>
) : ScheduledUpdater {

  /**
   * The size of the currently processing sequence (set at the first forward state).
   */
  var sequenceSize: Int = 0
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
  val stateEncodersPool: AttentionNetworksPool<DenseNDArray> =
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
   * The zeros array used as encoding of the initial state.
   */
  val initialStateEncoding = DenseNDArrayFactory.zeros(Shape(this.model.recurrentContextSize))

  /**
   * The list of transform layers groups used during the last forward.
   */
  val usedTransformLayers = mutableListOf<List<FeedforwardLayerStructure<DenseNDArray>>>()

  /**
   * The list of Attention Networks used to encode all the states of the current input sequence.
   */
  val usedStateEncoders = mutableListOf<AttentionNetwork<DenseNDArray>>()

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

    return this.forwardHelper.forward(firstState = firstState, sequence = sequence, lastPrediction = lastPrediction)
  }

  /**
   * Back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: List<DenseNDArray>) = this.backwardHelper.backward(outputErrors = outputErrors)

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
