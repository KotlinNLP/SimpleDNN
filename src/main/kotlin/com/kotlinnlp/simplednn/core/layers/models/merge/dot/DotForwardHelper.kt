/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.dot

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray


/**
 * The helper which executes the forward on a DotLayer.
 *
 * @property layer the layer in which the forward is executed
 */
class DotForwardHelper(override val layer: DotLayer) : ForwardHelper<DenseNDArray>(layer){

  /**
   * Forward the input to the output calculating the dot product between input1 and input2.
   */
  override fun forward() {
    this.layer.outputArray.assignValues(this.layer.inputArray1.values.t.dot(this.layer.inputArray2.values))
  }
}