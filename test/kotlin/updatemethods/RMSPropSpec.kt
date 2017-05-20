/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package updatemethods

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.rmsprop.RMSPropMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.rmsprop.RMSPropStructure
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class RMSPropSpec: Spek({

  describe("the RMSProp update method") {

    on("get support structure") {

      val updateHelper = RMSPropMethod(learningRate = 0.001, epsilon = 1e-06, decay = 0.9)
      val updateableArray = Utils.buildUpdateableArray()

      it("should return a support structure of the expected type") {
        assertEquals(true, updateHelper.getSupportStructure(updateableArray) is RMSPropStructure)
      }
    }

    on("update") {

      val updateHelper = RMSPropMethod(learningRate = 0.001, epsilon = 1e-06, decay = 0.9)
      val updateableArray = Utils.buildUpdateableArray()
      val supportStructure = updateHelper.getSupportStructure(updateableArray) as RMSPropStructure

      supportStructure.secondOrderMoments.assignValues(Utils.supportArray2())

      updateHelper.update(array = updateableArray, errors = Utils.buildErrors())

      it("should match the expected updated array") {
        assertEquals(true, updateableArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(0.399091, 0.398905, 0.499502, 0.996838, 0.799765)),
          tolerance = 1.0e-6))
      }
    }
  }
})
