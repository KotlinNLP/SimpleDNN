/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.embeddings.lmdb.EmbeddingsStorage

/**
 * TODO: Add documentation
 */
fun main(args: Array<String>) = EmbeddingsStorage(args[0], readOnly = false).use {
  it.load(args[1], verbose = true)
}