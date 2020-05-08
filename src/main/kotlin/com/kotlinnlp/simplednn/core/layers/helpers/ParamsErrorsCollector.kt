/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.helpers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList

/**
 * Generic params errors collector.
 */
class ParamsErrorsCollector {

  /**
   * The structure in which to accumulate the parameters errors.
   */
  private val paramsErrorsMap = mutableMapOf<String, ParamsArray.Errors<*>>()

  /**
   * Return the errors associated to the given [params].
   * If the [params] has no errors yet, these are built using the [params] default errors type (sparse vs. dense).
   *
   * @param params the parameters
   *
   * @return the current errors of the given parameters
   */
  fun getErrors(params: ParamsArray) =
    this.paramsErrorsMap.getOrPut(params.uuid, defaultValue = { params.buildDefaultErrors() })

  /**
   * @return all the collected params errors
   */
  fun getAll(): ParamsErrorsList = this.paramsErrorsMap.values.toList()

  /**
   * Clear the accumulated errors.
   */
  fun clear() = this.paramsErrorsMap.clear()
}
