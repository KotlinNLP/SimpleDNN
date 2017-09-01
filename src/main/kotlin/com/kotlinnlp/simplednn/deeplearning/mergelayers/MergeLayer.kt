/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.mergelayers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Merge Layer abstract class.
 * It is a [LayerStructure] with two inputs instead of one.
 *
 * @property inputArray1 the first input array of the layer
 * @property inputArray2 the second input array of the layer
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default 0.0).
 *                   If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
abstract class MergeLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArray1: AugmentedArray<InputNDArrayType>,
  val inputArray2: AugmentedArray<InputNDArrayType>,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: MergeLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double = 0.0
) : LayerStructure<InputNDArrayType>(
  inputArray = inputArray1,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout) {

  /**
   * @return the [MergeLayerParameters] used to store errors, compatible with this type of [MergeLayer]
   */
  abstract fun parametersErrorsFactory(): MergeLayerParameters
}
