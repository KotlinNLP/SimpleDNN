/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package decaymethods

import com.kotlinnlp.simplednn.core.functionalities.decaymethods.ExponentialDecay
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class ExponentialDecaySpec : Spek({

  describe("an Exponential decay method") {

    val decayMethod = ExponentialDecay(totalIterations = 10, initLearningRate = 0.01, finalLearningRate = 0.001)

    on("update with t=1") {
      it("should return the expected value") {
        assertEquals(0.01, decayMethod.update(learningRate = 0.01, t = 1))
      }
    }

    on("update with t=2") {
      it("should return the expected value") {
        assertEquals(0.007742636826811276, decayMethod.update(learningRate = 0.01, t = 2))
      }
    }

    on("update with t=3") {
      it("should return the expected value") {
        assertEquals(0.005994842503189416, decayMethod.update(learningRate = 0.007742636826811276, t = 3))
      }
    }

    on("update with t>1 and learningRate = finalLearningRate") {
      it("should return the expected value") {
        assertEquals(0.001, decayMethod.update(learningRate = 0.001, t = 10))
      }
    }
  }
})
