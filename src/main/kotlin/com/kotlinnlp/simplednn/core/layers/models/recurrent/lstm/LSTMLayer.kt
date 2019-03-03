/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 * The LSTM Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class LSTMLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters<*>,
  layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : GatedRecurrentLayer<InputNDArrayType>(
  inputArray = inputArray,
  outputArray = outputArray,
  params = params,
  layerContextWindow = layerContextWindow,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   *
   */
  val inputGate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val outputGate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val forgetGate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val candidate = RecurrentLayerUnit<InputNDArrayType>(outputArray.size)

  /**
   *
   */
  val cell = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(outputArray.size)))

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = LSTMForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = LSTMBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper = LSTMRelevanceHelper(layer = this)

  /**
   * Initialization: set the activation function of the gates
   */
  init {

    this.inputGate.setActivation(Sigmoid())
    this.outputGate.setActivation(Sigmoid())
    this.forgetGate.setActivation(Sigmoid())

    if (activationFunction != null) {
      this.candidate.setActivation(activationFunction)
      this.cell.setActivation(activationFunction)
    }
  }

  /**
   * Set the initial hidden array.
   * This method should be used when this layer is used as initial hidden state in a recurrent neural network.
   *
   * @param array the initial hidden array
   */
  override fun setInitHidden(array: DenseNDArray) {
    this.cell.values.zeros()
    this.outputArray.assignValues(array)
  }

  /**
   * Get the errors of the initial hidden array.
   * This method should be used only if this layer is used as initial hidden state in a recurrent neural network.
   *
   * @return the errors of the initial hidden array
   */
  override fun getInitHiddenErrors(): DenseNDArray =
    this.backwardHelper.getLayerRecurrentContribution(
      nextStateLayer = this.layerContextWindow.getNextState() as LSTMLayer<*>).t
}
