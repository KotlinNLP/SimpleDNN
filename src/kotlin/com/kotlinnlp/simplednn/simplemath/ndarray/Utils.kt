/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

typealias Indices = Pair<Int, Int>
typealias SparseEntry = Pair<Indices, Double>
typealias VectorIndices = ArrayList<Int>
typealias VectorsMap = MutableMap<Int, VectorIndices?>
typealias VectorsMapEntry = MutableMap.MutableEntry<Int, VectorIndices?>
