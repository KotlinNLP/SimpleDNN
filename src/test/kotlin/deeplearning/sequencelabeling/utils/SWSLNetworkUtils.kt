/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.sequencelabeling.utils

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.deeplearning.embeddings.Embedding
import com.kotlinnlp.simplednn.deeplearning.sequencelabeling.SWSLNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object SWSLNetworkUtils {

  /**
   *
   */
  fun buildNetwork(): SWSLNetwork {

    val network = SWSLNetwork(
      elementSize = 2,
      hiddenLayerSize = 5,
      hiddenLayerActivation = ELU(),
      numberOfLabels = 10,
      leftContextSize = 3,
      rightContextSize = 3,
      labelEmbeddingSize = 2)

    network.labelsEmbeddings.set(
      key = 0,
      embedding = Embedding(id = 0, array = UpdatableDenseArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, -0.9))))
    )

    return network
  }
}
