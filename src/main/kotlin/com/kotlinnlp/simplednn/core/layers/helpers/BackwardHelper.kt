/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.Layer
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
  private var paramsErrorsCollector = ParamsErrorsCollector()

  /**
   * The set of errors parameters which were considered during the last [backward].
   *
   * Note: the access to the errors is sufficient to include them in this set, even if they were not used in the
   * calculations really.
   */
  private val touchedErrors = mutableSetOf<ParamsArray.Errors<*>>()

  /**
   * Replace the current [paramsErrorsCollector] with [c].
   *
   * @param c a collector of params errors
   */
  fun setParamsErrorsCollector(c: ParamsErrorsCollector) { this.paramsErrorsCollector = c }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   *
   * @return the list of params errors
   */
  fun backward(propagateToInput: Boolean = false): ParamsErrorsList {

    this.touchedErrors.clear()

    this.execBackward(propagateToInput)

    return this.touchedErrors.toList()
  }

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  protected abstract fun execBackward(propagateToInput: Boolean = false)

  /**
   * Extension that allows you to access the errors of the parameters in the same way as you access the errors of
   * the layers.
   *
   * Given for example the params:
   *
   *    'this.layer.params.weights'
   *
   * you can access its local errors with:
   *
   *     'this.layer.params.weights.errors'
   *
   * The errors of this [ParamsArray] are extracted from the [paramsErrorsCollector]. If the parameter has no errors
   * yet, they are automatically created, initialized to zero, and returned.
   */
  val ParamsArray.errors: ParamsArray.Errors<*> get() {

    val errors = paramsErrorsCollector.getErrors(this)

    touchedErrors.add(errors)

    return errors
  }
}
