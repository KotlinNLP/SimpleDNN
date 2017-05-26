/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent

import com.kotlinnlp.simplednn.core.layers.LayerStructure

/**
 *
 */
interface LayerContextWindow {
  
  /**
   *
   */
  fun getPrevStateLayer(): LayerStructure<*>?

  /**
   *
   */
  fun getNextStateLayer(): LayerStructure<*>?
}
