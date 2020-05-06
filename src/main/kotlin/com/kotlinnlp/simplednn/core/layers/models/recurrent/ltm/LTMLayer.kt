/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentRelevanceHelper
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The LTM Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class LTMLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: LTMLayerParameters,
  layerContextWindow: LayerContextWindow,
  dropout: Double = 0.0
) : GatedRecurrentLayer<InputNDArrayType>(
  inputArray = inputArray,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  layerContextWindow = layerContextWindow,
  activationFunction = null,
  dropout = dropout) {

  /**
   * The input of the gates L1, L2, L3.
   * It is a support variable used during the calculations.
   */
  lateinit var x: NDArray<*>

  /**
   * The input of the cell.
   * It is a support variable used during the calculations.
   */
  val c: AugmentedArray<DenseNDArray> = AugmentedArray.zeros(this.inputArray.size)

  /**
   * The L1 input gate.
   */
  val inputGate1: AugmentedArray<DenseNDArray> = AugmentedArray.zeros(this.inputArray.size)

  /**
   * The L2 input gate.
   */
  val inputGate2: AugmentedArray<DenseNDArray> = AugmentedArray.zeros(this.inputArray.size)

  /**
   * The L3 input gate.
   */
  val inputGate3: AugmentedArray<DenseNDArray> = AugmentedArray.zeros(this.inputArray.size)

  /**
   * The cell.
   */
  val cell: AugmentedArray<DenseNDArray> = AugmentedArray.zeros(this.inputArray.size)

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = LTMForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = LTMBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: GatedRecurrentRelevanceHelper? = null

  /**
   * Initialization: set the activation function of the gates.
   */
  init {
    this.inputGate1.setActivation(Sigmoid)
    this.inputGate2.setActivation(Sigmoid)
    this.inputGate3.setActivation(Sigmoid)
    this.cell.setActivation(Sigmoid)
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
    (this.layerContextWindow.getNextState() as LTMLayer<*>).inputArray.errors
}
