/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
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
class LearningRateSpec: Spek({

  describe("the Learning Rate update method") {

    context("update with dense errors") {

      context("update") {

        val updateHelper = LearningRateMethod(learningRate = 0.001)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3991, 0.3993, 0.4996, 0.9992, 0.7999)),
              tolerance = 1.0e-6)
          }
        }
      }
    }

    context("update with sparse errors") {

      context("update") {

        val updateHelper = LearningRateMethod(learningRate = 0.001)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.3993, 0.5, 1.0, 0.7997)),
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

        val updateHelper = LearningRateMethod(learningRate = 0.001, decayMethod = decayMethod)

        updateHelper.newEpoch()

        it("should match the expected alpha in the first epoch") {
          assertEquals(0.03, updateHelper.alpha)
        }
      }

      context("second epoch") {

        val updateHelper = LearningRateMethod(learningRate = 0.001, decayMethod = decayMethod)

        updateHelper.newEpoch()
        updateHelper.newEpoch()

        it("should match the expected alpha in the second epoch") {
          assertEquals(0.05, updateHelper.alpha)
        }
      }
    }
  }
})
