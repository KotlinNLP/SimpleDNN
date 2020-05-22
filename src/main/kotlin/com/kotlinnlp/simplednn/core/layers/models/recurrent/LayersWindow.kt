/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.layers.Layer

/**
 * A window of recurrent layers.
 * It provides methods to get the current stacked layer in the next and the previous state of a recurrent network.
 * Useful during the forward and backward operations.
 */
internal interface LayersWindow {

  /**
   * @return the current layer in previous state
   */
  fun getPrevState(): Layer<*>?

  /**
   * @return the current layer in next state
   */
  fun getNextState(): Layer<*>?
}
