/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.decaymethods

import com.kotlinnlp.simplednn.core.functionalities.decaymethods.ExponentialDecay
import com.kotlinnlp.simplednn.simplemath.equals
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class ExponentialDecaySpec : Spek({

  describe("an Exponential decay method") {

    val decayMethod = ExponentialDecay(totalIterations = 10, initLearningRate = 0.01, finalLearningRate = 0.001)

    on("update with t=1") {
      it("should return the expected value") {
        assertTrue(equals(0.01, decayMethod.update(learningRate = 0.01, timeStep = 1), tolerance = 1.0e-08))
      }
    }

    on("update with t=2") {
      it("should return the expected value") {
        assertTrue(equals(0.00774264, decayMethod.update(learningRate = 0.01, timeStep = 2), tolerance = 1.0e-08))
      }
    }

    on("update with t=3") {
      it("should return the expected value") {
        assertTrue(equals(0.00599484, decayMethod.update(learningRate = 0.0077426368, timeStep = 3), tolerance = 1.0e-08))
      }
    }

    on("update with t>1 and learningRate = finalLearningRate") {
      it("should return the expected value") {
        assertTrue(equals(0.001, decayMethod.update(learningRate = 0.001, timeStep = 10), tolerance = 1.0e-08))
      }
    }
  }
})
