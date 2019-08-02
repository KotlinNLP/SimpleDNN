/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.ItemsPool

/**
 * The Attention Mechanism Layer Structure.
 *
 * @property inputArrays the input arrays of the layer
 * @param inputType the input array type (default Dense)
 * @param params the parameters which connect the input to the output
 * @param activation the activation function of the layer (default SoftmaxBase)
 * @param dropout the probability of dropout (default 0.0).
 *                If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class AttentionMechanismLayer(
  val inputArrays: List<AugmentedArray<DenseNDArray>>,
  inputType: LayerType.Input,
  params: LayerParameters,
  activation: ActivationFunction? = SoftmaxBase(),
  dropout: Double = 0.0,
  override val id: Int = 0
) : ItemsPool.IDItem,
  Layer<DenseNDArray>(
    inputArray = inputArrays[0],
    inputType = inputType,
    outputArray = AugmentedArray(inputArrays.size),
    params = params,
    activationFunction = activation,
    dropout = dropout
  ) {

  /**
   * A matrix containing the attention arrays as rows.
   */
  internal val attentionMatrix: AugmentedArray<DenseNDArray> = AugmentedArray(
    DenseNDArrayFactory.arrayOf(this.inputArrays.map { it.values.toDoubleArray() })
  )

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = AttentionMechanismForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = AttentionMechanismBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Initialization: set the activation function of the outputArray
   */
  init {

    require(this.inputArrays.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.inputArrays.all { it.values.length == (this.params as AttentionMechanismLayerParameters).inputSize }) {
      "The input arrays must have the expected size (%d).".format((this.params as AttentionMechanismLayerParameters).inputSize)
    }

    if (activationFunction != null) {
      this.outputArray.setActivation(activationFunction)
    }
  }
}
