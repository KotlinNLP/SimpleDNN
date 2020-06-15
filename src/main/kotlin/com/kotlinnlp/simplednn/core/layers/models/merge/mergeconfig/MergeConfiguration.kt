/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig

import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * The configuration of a merge layer.
 *
 * @property type the connection type
 */
abstract class MergeConfiguration(val type: LayerType.Connection) {

  /**
   * Check the connection type.
   */
  init {
    require(this.type.property == LayerType.Property.Merge)
  }
}
