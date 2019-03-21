/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.ParamsArray

/**
 * Generic params errors collector.
 */
class GenericParamsErrorsCollector {

  /**
   * The structure in which to accumulate the parameters errors.
   */
  private val paramsErrorsMap = mutableMapOf<String, ParamsArray.Errors<*>>()

  /**
   * Return the current errors of this parameters.
   */
  fun getErrors(params: ParamsArray) =
    this.paramsErrorsMap.getOrPut(params.uuid, defaultValue = { params.buildDenseErrors() } )

  /**
   * Clear the accumulated errors.
   */
  fun clear() = this.paramsErrorsMap.clear()
}
