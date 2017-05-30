/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package updatemethods

import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMStructure
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class ADAMSpec : Spek({

  describe("the ADAM update method") {

    context("update with dense errors") {

      on("get support structure") {

        val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
        val updatableArray: UpdatableDenseArray = Utils.buildUpdateableArray()

        it("should return a support structure of the expected type") {
          assertEquals(true, updateHelper.getSupportStructure(updatableArray) is ADAMStructure)
        }
      }

      on("update") {

        val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
        val updatableArray: UpdatableDenseArray = Utils.buildUpdateableArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray) as ADAMStructure

        supportStructure.firstOrderMoments.assignValues(Utils.supportArray1())
        supportStructure.secondOrderMoments.assignValues(Utils.supportArray2())

        updateHelper.update(array = updatableArray, errors = Utils.buildDenseErrors())

        it("should match the expected updated array") {
          assertEquals(true, updatableArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.39928, 0.39875, 0.49941, 0.98617, 0.79958)),
            tolerance = 1.0e-5))
        }
      }
    }

    context("update with sparse errors") {

      on("update") {

        val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)
        val updatableArray: UpdatableDenseArray = Utils.buildUpdateableArray()
        val supportStructure = updateHelper.getSupportStructure(updatableArray) as ADAMStructure

        supportStructure.firstOrderMoments.assignValues(Utils.supportArray1())
        supportStructure.secondOrderMoments.assignValues(Utils.supportArray2())

        updateHelper.update(array = updatableArray, errors = Utils.buildSparseErrors())

        it("should match the expected updated array") {
          assertEquals(true, updatableArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.398751, 0.5, 1.0, 0.79953)),
            tolerance = 1.0e-6))
        }
      }
    }

    on("newExample") {

      val updateHelper = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8)

      updateHelper.newExample()

      it("should update the 'alpha' parameter") {
        assertEquals(true, equals(updateHelper.alpha, 3.1623e-4, tolerance = 1.0e-08))
      }
    }
  }
})
