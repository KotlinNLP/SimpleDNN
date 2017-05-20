/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package updatemethods

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateStructure
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class LearningRateSpec: Spek({

  describe("the Learning Rate update method") {

    on("get support structure") {

      val updateHelper = LearningRateMethod(learningRate = 0.001)
      val updateableArray = Utils.buildUpdateableArray()

      it("should return a support structure of the expected type") {
        assertEquals(true, updateHelper.getSupportStructure(updateableArray) is LearningRateStructure)
      }
    }

    on("update") {

      val updateHelper = LearningRateMethod(learningRate = 0.001)
      val updateableArray = Utils.buildUpdateableArray()

      updateHelper.update(array = updateableArray, errors = Utils.buildErrors())

      it("should match the expected updated array") {
        assertEquals(true, updateableArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.3991, 0.3993, 0.4996, 0.9992, 0.7999)),
          tolerance = 1.0e-5))
      }
    }
  }
})
