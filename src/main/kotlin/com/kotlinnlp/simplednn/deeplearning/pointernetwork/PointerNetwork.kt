/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionStructure
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayerStructure
import com.kotlinnlp.simplednn.core.mergelayers.affine.AffineLayersPool
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [PointerNetwork].
 *
 * @property model the model of the network
 */
class PointerNetwork(val model: PointerNetworkModel) {

  /**
   * The processor of the recurrent network.
   */
  val recurrentProcessor: RecurrentNeuralProcessor<DenseNDArray> =
    RecurrentNeuralProcessor(this.model.recurrentNetwork)

  /**
   * A boolean indicating if the current is the first state.
   */
  private var firstState: Boolean = true

  /**
   * The input sequence that must be set using the [setEncodedSequence] method.
   */
  private lateinit var encodedSequence: List<DenseNDArray>

  /**
   * A pool of Feedforward Layers used to build the attention arrays.
   */
  val transformLayersPool: AffineLayersPool<DenseNDArray> =
    AffineLayersPool(
      inputType = LayerType.Input.Dense,
      activationFunction = Tanh(),
      params = this.model.transformParams)

  /**
   * The list of transform layers groups used during the last forward.
   */
  val usedTransformLayers = mutableListOf<List<AffineLayerStructure<DenseNDArray>>>()

  /**
   * The list of attention structures used during the last forward.
   */
  val usedAttentionStructures = mutableListOf<AttentionStructure>()

  /**
   * The forward helper.
   */
  private val forwardHelper = ForwardHelper(network = this)

  /**
   * The backward helper.
   */
  private val backwardHelper = BackwardHelper(network = this)

  /**
   * Set the encoded sequence.
   *
   * @param encodedSequence the encoded sequence
   */
  fun setEncodedSequence(encodedSequence: List<DenseNDArray>) {

    this.firstState = true
    this.encodedSequence = encodedSequence
  }

  /**
   * Forward.
   *
   * @param input the input
   *
   * @return an array that contains the importance score for each element of the input sequence
   */
  fun forward(input: DenseNDArray): DenseNDArray {

    if (this.firstState) this.resetHistory()

    val output: DenseNDArray = this.forwardHelper.forward(
      input = input,
      firstState = this.firstState,
      encodedSequence = this.encodedSequence)

    this.firstState = false

    return output
  }

  /**
   * Back-propagation of the errors.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    this.backwardHelper.backward(outputErrors = outputErrors)
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of this network
   */
  fun getParamsErrors(copy: Boolean = true): PointerNetworkParameters =
    this.backwardHelper.getParamsErrors(copy = copy)

  /**
   * @return the errors of the sequence
   */
  fun getInputSequenceErrors(): List<DenseNDArray> = this.backwardHelper.inputSequenceErrors

  /**
   * Reset the decoding structure of the network.
   */
  private fun resetHistory(){

    this.transformLayersPool.releaseAll()
    this.usedTransformLayers.clear()
    this.usedAttentionStructures.clear()
  }
}
