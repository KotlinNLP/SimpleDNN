/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.affine

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [AffineLayer]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default: 0.0 - if applying it, the usual value is 0.25)
 */
class AffineLayersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val params: AffineLayerParameters,
  val activationFunction: ActivationFunction?,
  val dropout: Double = 0.0
) : ItemsPool<AffineLayer<InputNDArrayType>>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [AffineLayer] with the given [id]
   */
  override fun itemFactory(id: Int): AffineLayer<InputNDArrayType> {

    val inputArrays: List<AugmentedArray<InputNDArrayType>> =
      List(size = this.params.inputsSize.size, init = {
        AugmentedArray<InputNDArrayType>(size = this.params.inputsSize[it])
      })

    return AffineLayer(
      inputArrays = inputArrays,
      outputArray = AugmentedArray.zeros(this.params.outputSize),
      params = this.params,
      activationFunction = this.activationFunction,
      dropout = this.dropout,
      id = id
    )
  }
}
