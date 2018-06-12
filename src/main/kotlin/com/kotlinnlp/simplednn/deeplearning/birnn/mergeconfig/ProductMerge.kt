/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.mergeconfig

import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * A data class that defines that the output Merge layer of a [com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN] is a
 * Product layer.
 *
 * @property dropout the probability of dropout
 */
class ProductMerge(dropout: Double = 0.0) : MergeConfiguration(type = LayerType.Connection.Product, dropout = dropout)
