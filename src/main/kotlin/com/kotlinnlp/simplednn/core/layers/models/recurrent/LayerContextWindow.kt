/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent

import com.kotlinnlp.simplednn.core.layers.Layer

/**
 * The context window for a recurrent layer. It permits to get the layer in the previous and next states.
 */
interface LayerContextWindow {
  
  /**
   * @return the current layer in previous state
   */
  fun getPrevState(): Layer<*>?

  /**
   * @return the current layer in next state
   */
  fun getNextState(): Layer<*>?
}
