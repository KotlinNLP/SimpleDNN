/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.arrays

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
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

        it("should contain a support structure initialized with null") {
          assertNull(paramsArray.updaterSupportStructure)
        }

        it("should have a different uuid of the one of another instance") {
          assertNotEquals(paramsArray.uuid, paramsArray2.uuid)
        }
      }

      on("with a DoubleArray") {

        val paramsArray = ParamsArray(doubleArrayOf(0.3, 0.4, 0.2, -0.2))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4, 0.2, -0.2)))
        }
      }

      on("with a list of DoubleArray") {

        val paramsArray = ParamsArray(listOf(
          doubleArrayOf(0.3, 0.4, 0.2, -0.2),
          doubleArrayOf(0.2, -0.1, 0.1, 0.6)
        ))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.3, 0.4, 0.2, -0.2),
            doubleArrayOf(0.2, -0.1, 0.1, 0.6)
          )))
        }
      }

      on("with a matrix shape and initialized values") {

        val paramsArray = ParamsArray(Shape(2, 4), initializer = ConstantInitializer(0.42))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.42, 0.42, 0.42, 0.42),
            doubleArrayOf(0.42, 0.42, 0.42, 0.42)
          )))
        }
      }

      on("with a vector shape and initialized values") {

        val paramsArray = ParamsArray(size = 4, initializer = ConstantInitializer(0.42))

        it("should contain the expected values") {
          assertEquals(paramsArray.values, DenseNDArrayFactory.arrayOf(
            doubleArrayOf(0.42, 0.42, 0.42, 0.42)
          ))
        }
      }
    }

    on("support structure") {

      val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7))).apply {
        getOrSetSupportStructure<LearningRateStructure>()
      }

      it("should have the expected support structure type") {
        assertTrue { paramsArray.updaterSupportStructure is LearningRateStructure }
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

      on("build with default sparse errors") {

        val paramsArray = ParamsArray(
          values = DenseNDArrayFactory.zeros(Shape(3, 7)),
          defaultErrorsType = ParamsArray.ErrorsType.Sparse)

        val paramsErrors = paramsArray.buildDefaultErrors()

        it("should create sparse errors"){
          assertTrue { paramsErrors.values is SparseNDArray }
        }
      }
    }

    on("build with default dense errors") {

      val paramsArray = ParamsArray(
        values = DenseNDArrayFactory.zeros(Shape(3, 7)),
        defaultErrorsType = ParamsArray.ErrorsType.Dense)

      val paramsErrors = paramsArray.buildDefaultErrors()

      it("should create dense errors"){
        assertTrue { paramsErrors.values is DenseNDArray }
      }
    }

    on("build with default errors") {

      val paramsArray = ParamsArray(DenseNDArrayFactory.zeros(Shape(3, 7)))

      val paramsErrors = paramsArray.buildDefaultErrors()

      it("should create dense errors"){
        assertTrue { paramsErrors.values is DenseNDArray }
      }
    }
  }
})
