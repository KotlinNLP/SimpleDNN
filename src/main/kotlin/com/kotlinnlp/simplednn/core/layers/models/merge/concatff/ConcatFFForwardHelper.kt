/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concatff

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [ConcatFFLayer].
 *
 * @property layer the layer in which the forward is executed
 */
internal class ConcatFFForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: ConcatFFLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output through a concatenation and a feed-forward layer.
   * TODO: make it working also with non-dense input arrays.
   */
  override fun forward() {

    this.layer.outputFeedforward.inputArray.assignValues(
      concatVectorsV(this.layer.inputArrays.map { it.values as DenseNDArray })
    )

    this.layer.outputFeedforward.forward()
  }
}
