/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

internal typealias Indices = Pair<Int, Int>
internal typealias SparseEntry = Pair<Indices, Double>
internal typealias VectorIndices = MutableList<Int>
internal typealias VectorsMap = MutableMap<Int, VectorIndices?>
internal typealias VectorsMapEntry = MutableMap.MutableEntry<Int, VectorIndices?>
