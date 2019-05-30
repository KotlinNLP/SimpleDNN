/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.momentum.MomentumMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.nhaarman.mockito_kotlin.any
import com.nhaarman.mockito_kotlin.eq
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class MomentumSpec: Spek({

  describe("the Momentum update method") {

    context("update with dense errors") {

      context("update") {

        val updateHelper = MomentumMethod(learningRate = 0.001, momentum = 0.9)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.v.assignValues(UpdateMethodsUtils.supportArray1())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2309, -0.3207, 0.0496, 0.7292, 0.6199)),
              tolerance = 1.0e-6)
          }
        }
      }
    }

    context("update with sparse errors") {

      context("update") {

        val updateHelper = MomentumMethod(learningRate = 0.001, momentum = 0.9)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.v.assignValues(UpdateMethodsUtils.supportArray1())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.23, -0.3207, 0.05, 0.73, 0.6197)),
              tolerance = 1.0e-5)
          }
        }
      }
    }

    context("epoch scheduling") {

      val decayMethod = mock<DecayMethod>()
      whenever(decayMethod.update(learningRate = any(), timeStep = eq(1))).thenReturn(0.03)
      whenever(decayMethod.update(learningRate = any(), timeStep = eq(2))).thenReturn(0.05)

      context("first epoch") {

        val updateHelper = MomentumMethod(learningRate = 0.001, decayMethod = decayMethod)

        updateHelper.newEpoch()

        it("should match the expected alpha in the first epoch") {
          assertEquals(0.03, updateHelper.alpha)
        }
      }

      context("second epoch") {

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
