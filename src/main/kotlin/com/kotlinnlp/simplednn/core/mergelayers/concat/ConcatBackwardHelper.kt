/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.concat

import com.kotlinnlp.simplednn.core.layers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on an concat [layer].
 *
 * @property layer the [ConcatLayerStructure] in which the backward is executed
 */
class ConcatBackwardHelper(override val layer: ConcatLayerStructure) : BackwardHelper<DenseNDArray> {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
   *                the top errors (ignored if null)
   */
  override fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean, mePropK: Double?) {

    if (propagateToInput) {

      val gy: DenseNDArray = this.layer.outputArray.errors
      val part: Array<DenseNDArray> = gy.splitV(this.layer.inputArray1.size, this.layer.inputArray2.size)

      this.layer.inputArray1.assignErrors(part[0])
      this.layer.inputArray2.assignErrors(part[1])
    }
  }
}