/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.GatedRecurrentRelevanceHelper
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The DeltaRNN Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property layerContextWindow the context window used for the forward and the backward
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class DeltaRNNLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  params: LayerParameters,
  layerContextWindow: LayerContextWindow,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : GatedRecurrentLayer<InputNDArrayType>(
  inputArray = inputArray,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  layerContextWindow = layerContextWindow,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * The candidate array.
   */
  val candidate = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(this.outputArray.size)))

  /**
   * The partition array.
   */
  val partition = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(this.outputArray.size)))

  /**
   * The array which contains the result of the dot product between the input (x) and the input weights (w).
   */
  val wx = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(this.outputArray.size)))

  /**
   * The array which contains the result of the dot product between the output in the previous state and the recurrent
   * weights.
   */
  val wyRec = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(this.outputArray.size)))

  /**
   * The support structure used to save temporary results during a forward and using them to calculate the relevance
   * later.
   */
  val relevanceSupport: DeltaRNNRelevanceSupport
    get() = try {
      this._relevanceSupport

    } catch (e: UninitializedPropertyAccessException) {
      this._relevanceSupport = DeltaRNNRelevanceSupport(outputSize = this.outputArray.size)
      this._relevanceSupport
    }

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = DeltaRNNForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = DeltaRNNBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  @Suppress("UNCHECKED_CAST")
  override val relevanceHelper: GatedRecurrentRelevanceHelper? = if (this.denseInput)
    DeltaRNNRelevanceHelper(layer = this as DeltaRNNLayer<DenseNDArray>)
  else
    null

  /**
   * The support structure used to save temporary results during a forward and using them to calculate the relevance
   * later.
   */
  private lateinit var _relevanceSupport: DeltaRNNRelevanceSupport

  /**
   * Initialization: set the activation functions.
   */
  init {

    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }

    this.candidate.setActivation(activationFunction ?: Tanh())
    this.partition.setActivation(Sigmoid())
  }

  /**
   * Set the initial hidden array.
   * This method should be used when this layer is used as initial hidden state in a recurrent neural network.
   *
   * @param array the initial hidden array
   */
  override fun setInitHidden(array: DenseNDArray) {
    TODO("not implemented")
  }

  /**
   * Get the errors of the initial hidden array.
   * This method should be used only if this layer is used as initial hidden state in a recurrent neural network.
   *
   * @return the errors of the initial hidden array
   */
  override fun getInitHiddenErrors(): DenseNDArray {
    TODO("not implemented")
  }
}
