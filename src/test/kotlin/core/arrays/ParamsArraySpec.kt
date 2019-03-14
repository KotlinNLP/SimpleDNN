/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.arrays

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.*

/**
 *
 */
class ParamsArraySpec : Spek({

  describe("a ParamsArray") {

    context("initialization") {

      on("with an NDArray") {

        val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))
        val paramsArray2 = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

        it("should contain values with the expected number of rows") {
          assertEquals(3, paramsArray.values.rows)
        }

        it("should contain values with the expected number of columns") {
          assertEquals(7, paramsArray.values.columns)
        }

        it("should raise an Exception when trying to access its structure without setting it") {
          assertFailsWith<UninitializedPropertyAccessException> { paramsArray.updaterSupportStructure }
        }

        it("should have a different uuid of the one of another instance") {
          assertNotEquals(paramsArray.uuid, paramsArray2.uuid)
        }
      }

      on("from an UpdatableArray") {

        val updatableArray = UpdatableDenseArray(DenseNDArrayFactory.zeros(Shape(3, 7))).apply {
          getOrSetSupportStructure<LearningRateStructure>()
        }

        val paramsArray = ParamsArray(updatableArray)

        it("should have the same values as the updatable array") {
          assertSame(paramsArray.values, updatableArray.values)
        }

        it("should have the same support structure as the updatable array") {
          assertSame(paramsArray.updaterSupportStructure, updatableArray.updaterSupportStructure)
        }
      }
    }

    context("params errors") {

      on("build a dense errors without values") {

        val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))
        val paramsArray2 = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

        val paramsErrors = paramsArray.buildDenseErrors()

        it("should contain values with the expected shape") {
          assertEquals(paramsArray.values.shape, paramsErrors.values.shape)
        }

        it("should contain the expected values") {
          assertEquals(paramsErrors.values, DenseNDArrayFactory.zeros(Shape(3, 7)))
        }

        it("should contains the right reference to the paramsArray"){
          assertSame(paramsErrors.refParams, paramsArray)
        }

        it("shouldn't contains the reference to the paramsArray2"){
          assertNotSame(paramsErrors.refParams, paramsArray2)
        }

        it("should create its copy with the same reference to the paramsArray"){
          assertSame(paramsErrors.copy().refParams, paramsArray)
        }
      }
    }
  }
})
