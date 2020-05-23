/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.biaffine

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Biaffine Layer Structure.
 * WARNING: actually the forward and backward operations are NOT OPTIMIZED for sparse inputs.
 *
 * @property inputArray the first input array of the layer
 * @property inputArray2 the second input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout
 */
internal class BiaffineLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  internal val inputArray1: AugmentedArray<InputNDArrayType>,
  internal val inputArray2: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: BiaffineLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double
) : MergeLayer<InputNDArrayType>(
  inputArrays = listOf(inputArray1, inputArray2),
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout
) {

  /**
   * Constructor by params.
   *
   * @param params the parameters which connect the input to the output
   * @param activationFunction the activation function of the layer
   * @param dropout the probability of dropout
   */
  constructor(
    params: BiaffineLayerParameters,
    activationFunction: ActivationFunction? = null,
    dropout: Double
  ): this(
    inputArray1 = AugmentedArray<InputNDArrayType>(size = params.inputSize1),
    inputArray2 = AugmentedArray<InputNDArrayType>(size = params.inputSize2),
    inputType = if (params.sparseInput) LayerType.Input.SparseBinary else LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(size = params.outputSize),
    params = params,
    activationFunction = activationFunction,
    dropout = dropout
  )

  /**
   * A support structure used for calculations. Each array wx1i is obtained by: wi (dot) x1.
   */
  val wx1Arrays: List<DenseNDArray> = List(
    size = this.params.outputSize,
    init = { DenseNDArrayFactory.emptyArray(Shape(this.params.inputSize2)) }
  )

  /**
   * The helper which execute the forward.
   */
  override val forwardHelper = BiaffineForwardHelper(layer = this)

  /**
   * The helper which execute the backward.
   */
  override val backwardHelper = BiaffineBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Initialization: set the activation function of the outputArray.
   */
  init {
    if (activationFunction != null) {
      outputArray.setActivation(activationFunction)
    }
  }
}
