/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.han

/**
 * The HierarchyGroup represents a higher level of the input hierarchy of a [HANEncoder].
 *
 * It contains a list of other [HierarchyItem]s as sub-levels.
 */
class HierarchyGroup(vararg groups: HierarchyItem) : HierarchyItem, ArrayList<HierarchyItem>(groups.size) {

  init {
    groups.forEach {
      this.add(it)
    }
  }
}
