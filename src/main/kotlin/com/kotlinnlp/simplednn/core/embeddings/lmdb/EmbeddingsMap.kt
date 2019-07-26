/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.embeddings.lmdb

import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer

/**
 * @param storage the embeddings storage
 * @param initializer the initializer of the values of the embeddings (zeros if null, default: Glorot)
 * @param pseudoRandomDropout a Boolean indicating if embeddings must be dropped out with pseudo random probability
 *                            (default = true)
 */
class EmbeddingsMap(
  storage: EmbeddingsStorage,
  initializer: Initializer? = GlorotInitializer(),
  pseudoRandomDropout: Boolean = true
) : EmbeddingsMap<String>(
  size = storage.embeddingsSize,
  initializer = initializer,
  pseudoRandomDropout = pseudoRandomDropout
) {
  override val embeddings = storage
}
