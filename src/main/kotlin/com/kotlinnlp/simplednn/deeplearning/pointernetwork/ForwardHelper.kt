/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The forward helper of the [PointerNetwork].
 *
 * @property network the attentive recurrent network of this helper
 */
class ForwardHelper(private val network: PointerNetwork) {

  /**
   * The recurrent vector used in first state.
   */
  private val initRecurrentVector: DenseNDArray =
    DenseNDArrayFactory.zeros(Shape(this.network.model.recurrentHiddenSize))

  /**
   * A boolean indicating if the current is the first state of recursion.
   */
  private var firstRecurrentState: Boolean = true

  /**
   * @param inputSequence the input sequence
   * @param firstState a boolean indicating if this is the first state
   * @param initHidden the initial hidden array (null by default)
   */
  fun forward(inputSequence: List<DenseNDArray>,
              firstState: Boolean,
              initHidden: DenseNDArray? = null): DenseNDArray {

    TODO()
  }
}
