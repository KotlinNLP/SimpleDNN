/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.attention.han

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The HierarchySequence represents the lowest level of the input hierarchy of a [HANEncoder].
 *
 * It contains a list of [AugmentedArray]s as input sequence of the lowest level of a [HANEncoder].
 */
class HierarchySequence<NDArrayType: NDArray<NDArrayType>>(vararg arrays: NDArrayType)
  : HierarchyItem, ArrayList<NDArrayType>(arrays.size) {

  init {
    arrays.forEach {
      this.add(it)
    }
  }
}
