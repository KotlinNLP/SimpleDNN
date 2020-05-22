/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * A window of recurrent states.
 * It provides methods to get to the next and the previous state of a recurrent network.
 * Useful during the forward and backward operations.
 */
abstract class StatesWindow<InputNDArrayType : NDArray<InputNDArrayType>> {

  /**
   * @return the previous recurrent state
   */
  internal abstract fun getPrevState(): RecurrentStackedLayers<InputNDArrayType>?

  /**
   * @return the next recurrent state
   */
  internal abstract fun getNextState(): RecurrentStackedLayers<InputNDArrayType>?
}
