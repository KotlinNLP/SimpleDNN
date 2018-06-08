/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.mergeconfig

import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * A class that defines the configuration of the output Merge layer of a
 * [com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN].
 *
 * @property type the connection type of the output Merge layer
 */
abstract class MergeConfiguration(val type: LayerType.Connection) {

  /**
   * Check connection type.
   */
  init {
    require(this.type.property == LayerType.Property.Merge)
  }
}
