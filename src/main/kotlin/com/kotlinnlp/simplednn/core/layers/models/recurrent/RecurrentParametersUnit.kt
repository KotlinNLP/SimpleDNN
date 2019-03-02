/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.models.ParametersUnit

/**
 * The parameters associated to a [RecurrentLayerUnit].
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param sparseInput whether the weights connected to the input are sparse or not (default false)
 * @param meProp whether to use the 'meProp' errors propagation algorithm (params are sparse) (default false)
 */
class RecurrentParametersUnit(
  inputSize: Int,
  outputSize: Int,
  sparseInput: Boolean = false,
  meProp: Boolean = false
) : ParametersUnit(
  inputSize = inputSize,
  outputSize = outputSize,
  sparseInput = sparseInput) {

  /**
   *
   */
  val recurrentWeights: UpdatableArray<*> = UpdatableArray(
    dim1 = this.outputSize,
    dim2 = this.outputSize,
    sparse = meProp)
}
