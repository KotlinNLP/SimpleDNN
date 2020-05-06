/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.autoassociative

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.ItemsPool

/**
 * The Feedforward Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the output (can be null, default: Sigmoid)
 * @property lambda the partition factor for the reconstruction (default = 0.75)
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class NewRecirculationLayer(
  inputArray: AugmentedArray<DenseNDArray>,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: NewRecirculationLayerParameters,
  activationFunction: ActivationFunction? = Sigmoid,
  val lambda: Double = 0.75,
  dropout: Double = 0.0,
  override val id: Int = 0
) : ItemsPool.IDItem,
  Layer<DenseNDArray>(
    inputArray = inputArray,
    inputType = LayerType.Input.Dense,
    outputArray = outputArray,
    params = params,
    activationFunction = activationFunction,
    dropout = dropout
  ) {

  /**
   *
   */
  val realInput = inputArray

  /**
   *
   */
  val imaginaryInput = outputArray

  /**
   *
   */
  val realOutput = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.params.hiddenSize)))

  /**
   *
   */
  val imaginaryOutput = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(this.params.hiddenSize)))


  /**
   * The helper which executes the forward
   */
  override val forwardHelper = NewRecirculationForwardHelper(this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = NewRecirculationBackwardHelper(this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Initialization: set the activation function of the outputArray
   */
  init {

    require(this.inputArray.size == this.outputArray.size) {
      "The inputArray and the outputArray must have the same size."
    }

    if (activationFunction != null) {
      this.realOutput.setActivation(activationFunction)
      this.imaginaryOutput.setActivation(activationFunction)
    }
  }
}
