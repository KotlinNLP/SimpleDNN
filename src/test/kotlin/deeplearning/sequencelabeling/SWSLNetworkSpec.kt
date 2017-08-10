/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.sequencelabeling

import deeplearning.sequencelabeling.utils.SWSLNetworkUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class SWSLNetworkSpec: Spek({

  describe("an SWSLNetwork") {

    context("input=2, hidden=5, output=10, prevWindow=3, nextWindow=3, labelEmbedding=2") {

      val network = SWSLNetworkUtils.buildNetwork()

      on("featuresSize"){

        it("should be 20") {
          assertEquals(20, network.featuresSize)
        }
      }
    }
  }
})
