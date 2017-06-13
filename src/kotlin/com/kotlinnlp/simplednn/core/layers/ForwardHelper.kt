/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the forward on a [layer].
 */
interface ForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>> {

  /**
   * The [LayerStructure] in which the forward is executed.
   */
  val layer: LayerStructure<InputNDArrayType>

  /**
   * Forward the input to the output combining it with the parameters.
   */
  fun forward()

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  fun forward(paramsContributes: LayerParameters)
}
