/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package updatemethods

import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumStructure
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.nhaarman.mockito_kotlin.any
import com.nhaarman.mockito_kotlin.eq
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class MomentumSpec: Spek({

  describe("the Momentum update method") {

    on("get support structure") {

      val updateHelper = MomentumMethod(learningRate = 0.001, momentum = 0.9)
      val updateableArray = Utils.buildUpdateableArray()

      it("should return a support structure of the expected type") {
        assertEquals(true, updateHelper.getSupportStructure(updateableArray) is MomentumStructure)
      }
    }

    on("update") {

      val updateHelper = MomentumMethod(learningRate = 0.001, momentum = 0.9)
      val updateableArray = Utils.buildUpdateableArray()
      val supportStructure = updateHelper.getSupportStructure(updateableArray) as MomentumStructure

      supportStructure.v.assignValues(Utils.supportArray1())

      updateHelper.update(array = updateableArray, errors = Utils.buildErrors())

      it("should match the expected updated array") {
        assertEquals(true, updateableArray.values.equals(
            NDArray.arrayOf(doubleArrayOf(-0.2309, -0.3207, 0.0496, 0.7292, 0.6199)),
          tolerance = 1.0e-5))
      }
    }

    context("epoch scheduling") {

      val decayMethod = mock<DecayMethod>()
      whenever(decayMethod.update(learningRate = any(), timeStep = eq(1))).thenReturn(0.03)
      whenever(decayMethod.update(learningRate = any(), timeStep = eq(2))).thenReturn(0.05)

      on("first epoch") {

        val updateHelper = MomentumMethod(learningRate = 0.001, decayMethod = decayMethod)

        updateHelper.newEpoch()

        it("should match the expected alpha in the first epoch") {
          assertEquals(0.03, updateHelper.alpha)
        }
      }

      on("second epoch") {

        val updateHelper = MomentumMethod(learningRate = 0.001, decayMethod = decayMethod)

        updateHelper.newEpoch()
        updateHelper.newEpoch()

        it("should match the expected alpha in the second epoch") {
          assertEquals(0.05, updateHelper.alpha)
        }
      }
    }
  }
})
