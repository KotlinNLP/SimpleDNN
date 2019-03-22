/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.optimizer.GenericParamsErrorsCollector
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @param layer the layer in which the backward is executed
 */
abstract class BackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  protected open val layer: Layer<InputNDArrayType>
) {

  /**
   * The errors of the parameters which will be filled at each [backward].
   */
  private val paramsErrorsCollector = GenericParamsErrorsCollector()

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  fun backward(propagateToInput: Boolean = false): ParamsErrorsList {

    this.execBackward(propagateToInput)

    return this.paramsErrorsCollector.getAll()
  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  abstract fun execBackward(propagateToInput: Boolean = false)

  /**
   *
   */
  val ParamsArray.errors: ParamsArray.Errors<*> get() = paramsErrorsCollector.getErrors(this)
}
