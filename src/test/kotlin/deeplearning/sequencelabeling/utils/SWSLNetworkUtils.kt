/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.sequencelabeling.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.deeplearning.sequencelabeling.SWSLNetwork

/**
 *
 */
object SWSLNetworkUtils {

  /**
   *
   */
  fun buildNetwork() = SWSLNetwork(
    elementSize = 2,
    hiddenLayerSize = 5,
    hiddenLayerActivation = ELU(),
    numberOfLabels = 10,
    leftContextSize = 3,
    rightContextSize = 3,
    labelEmbeddingSize = 2)
}
