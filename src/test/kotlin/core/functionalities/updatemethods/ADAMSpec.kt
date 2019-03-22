/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class ADAMSpec : Spek({

  describe("the ADAM update method") {

    context("update with dense errors") {

      on("update") {

        val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
        supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildDenseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39928, 0.398751, 0.499414, 0.986165, 0.799575)),
              tolerance = 1.0e-6)
          }
        }
      }
    }

    context("update with sparse errors") {

      on("update") {

        val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
        val updatableArray: ParamsArray = UpdateMethodsUtils.buildParamsArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray)

        supportStructure.firstOrderMoments.assignValues(UpdateMethodsUtils.supportArray1())
        supportStructure.secondOrderMoments.assignValues(UpdateMethodsUtils.supportArray2())

        updateHelper.update(array = updatableArray, errors = UpdateMethodsUtils.buildSparseErrors())

        it("should match the expected updated array") {
          assertTrue {
            updatableArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39937, 0.398751, 0.499462, -26999.0, 0.79953)),
              tolerance = 1.0e-6)
          }
        }
      }
    }

    on("newExample") {

      val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)

      updateHelper.newExample()

      it("should update the 'alpha' parameter") {
        assertTrue { equals(updateHelper.alpha, 3.1623e-4, tolerance = 1.0e-08) }
      }
    }
  }
})
